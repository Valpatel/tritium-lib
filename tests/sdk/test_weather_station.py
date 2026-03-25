# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the Weather Station example addon — full SDK lifecycle coverage."""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from tritium_lib.sdk.examples.weather_station import (
    WeatherReading,
    WeatherSimulator,
    WeatherStationAddon,
)
from tritium_lib.sdk.addon_events import AddonEventBus
from tritium_lib.sdk.context import AddonContext
from tritium_lib.sdk.manifest import load_manifest, validate_manifest


# ---------------------------------------------------------------------------
# Test helpers — lightweight fakes for ITargetTracker and IEventBus
# ---------------------------------------------------------------------------

class FakeTargetTracker:
    """In-memory target tracker for testing."""

    def __init__(self) -> None:
        self.targets: dict[str, dict] = {}

    def update_target(self, target_id: str, data: dict) -> None:
        self.targets[target_id] = data

    def get_target(self, target_id: str) -> dict | None:
        return self.targets.get(target_id)

    def get_all_targets(self) -> list[dict]:
        return list(self.targets.values())

    def remove_target(self, target_id: str) -> bool:
        return self.targets.pop(target_id, None) is not None


class FakeEventBus:
    """In-memory event bus for testing."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any, str]] = []

    def publish(self, topic: str, data: Any = None, source: str = "") -> None:
        self.events.append((topic, data, source))

    def subscribe(self, topic: str, callback: Callable) -> None:
        pass


# ---------------------------------------------------------------------------
# WeatherReading tests
# ---------------------------------------------------------------------------

class TestWeatherReading:
    """Tests for the WeatherReading data model."""

    def test_default_values(self):
        r = WeatherReading(station_id="test")
        assert r.station_id == "test"
        assert r.temperature_c == 20.0
        assert r.humidity_pct == 50.0
        assert r.wind_speed_kph == 0.0
        assert r.pressure_hpa == 1013.25
        assert r.rain_mm_hr == 0.0
        assert r.visibility_km == 10.0

    def test_to_dict_serialization(self):
        r = WeatherReading(station_id="wx-1", temperature_c=25.123, humidity_pct=62.789)
        d = r.to_dict()
        assert d["station_id"] == "wx-1"
        assert d["temperature_c"] == 25.1  # rounded to 1 decimal
        assert d["humidity_pct"] == 62.8
        assert "timestamp" in d

    def test_temperature_f_conversion(self):
        r = WeatherReading(station_id="t", temperature_c=0.0)
        assert r.temperature_f == 32.0
        r2 = WeatherReading(station_id="t", temperature_c=100.0)
        assert r2.temperature_f == 212.0

    def test_wind_chill_applies_when_cold_and_windy(self):
        r = WeatherReading(station_id="t", temperature_c=5.0, wind_speed_kph=20.0)
        # Wind chill should be lower than actual temp
        assert r.wind_chill_c < r.temperature_c

    def test_wind_chill_returns_temp_when_warm(self):
        r = WeatherReading(station_id="t", temperature_c=15.0, wind_speed_kph=20.0)
        assert r.wind_chill_c == 15.0  # Too warm for wind chill

    def test_wind_chill_returns_temp_when_calm(self):
        r = WeatherReading(station_id="t", temperature_c=0.0, wind_speed_kph=2.0)
        assert r.wind_chill_c == 0.0  # Too calm for wind chill

    def test_heat_index_applies_when_hot(self):
        r = WeatherReading(station_id="t", temperature_c=35.0, humidity_pct=80.0)
        # Heat index should be higher than actual temp in hot humid conditions
        assert r.heat_index_c > r.temperature_c

    def test_heat_index_returns_temp_when_cool(self):
        r = WeatherReading(station_id="t", temperature_c=20.0, humidity_pct=50.0)
        assert r.heat_index_c == 20.0  # Too cool for heat index

    def test_classify_clear(self):
        r = WeatherReading(station_id="t")
        assert r.classify_conditions() == "clear"

    def test_classify_heavy_rain(self):
        r = WeatherReading(station_id="t", rain_mm_hr=10.0)
        assert r.classify_conditions() == "heavy_rain"

    def test_classify_moderate_rain(self):
        r = WeatherReading(station_id="t", rain_mm_hr=5.0)
        assert r.classify_conditions() == "moderate_rain"

    def test_classify_light_rain(self):
        r = WeatherReading(station_id="t", rain_mm_hr=1.0)
        assert r.classify_conditions() == "light_rain"

    def test_classify_fog(self):
        r = WeatherReading(station_id="t", visibility_km=0.5)
        assert r.classify_conditions() == "fog"

    def test_classify_storm(self):
        r = WeatherReading(station_id="t", wind_speed_kph=70.0)
        assert r.classify_conditions() == "storm"

    def test_classify_windy(self):
        r = WeatherReading(station_id="t", wind_speed_kph=40.0)
        assert r.classify_conditions() == "windy"

    def test_classify_extreme_heat(self):
        r = WeatherReading(station_id="t", temperature_c=40.0)
        assert r.classify_conditions() == "extreme_heat"

    def test_classify_extreme_cold(self):
        r = WeatherReading(station_id="t", temperature_c=-15.0)
        assert r.classify_conditions() == "extreme_cold"

    def test_classify_humid(self):
        r = WeatherReading(station_id="t", humidity_pct=95.0)
        assert r.classify_conditions() == "humid"

    def test_severity_levels(self):
        assert WeatherReading(station_id="t").severity_level() == 0  # clear
        assert WeatherReading(station_id="t", rain_mm_hr=1.0).severity_level() == 1
        assert WeatherReading(station_id="t", rain_mm_hr=10.0).severity_level() == 2
        assert WeatherReading(station_id="t", wind_speed_kph=70.0).severity_level() == 3

    def test_rain_classification_priority(self):
        """Heavy rain is checked before fog, wind, etc."""
        r = WeatherReading(
            station_id="t",
            rain_mm_hr=10.0,
            visibility_km=0.5,
            wind_speed_kph=70.0,
        )
        assert r.classify_conditions() == "heavy_rain"


# ---------------------------------------------------------------------------
# WeatherSimulator tests
# ---------------------------------------------------------------------------

class TestWeatherSimulator:
    """Tests for the weather data simulator."""

    def test_creates_with_defaults(self):
        sim = WeatherSimulator()
        assert sim.station_id == "wx-sim-001"
        assert sim.lat == 39.7392
        assert sim.lng == -104.9903

    def test_read_returns_reading(self):
        sim = WeatherSimulator(station_id="test-sim")
        reading = sim.read()
        assert isinstance(reading, WeatherReading)
        assert reading.station_id == "test-sim"

    def test_readings_vary_over_time(self):
        """Readings at different times should differ."""
        sim = WeatherSimulator(station_id="ts")
        r1 = sim.read(sim_time=1000.0)
        r2 = sim.read(sim_time=50000.0)
        # Temperatures should differ due to day/night cycle
        assert r1.temperature_c != r2.temperature_c

    def test_temperature_in_reasonable_range(self):
        sim = WeatherSimulator(base_temp_c=20.0, temp_amplitude_c=8.0)
        for t in range(0, 86400, 3600):
            r = sim.read(sim_time=float(t))
            # With noise, allow wide range but should be reasonable
            assert -20.0 < r.temperature_c < 60.0

    def test_humidity_bounded(self):
        sim = WeatherSimulator()
        for _ in range(50):
            r = sim.read()
            assert 10.0 <= r.humidity_pct <= 100.0

    def test_pressure_bounded(self):
        sim = WeatherSimulator()
        for _ in range(100):
            r = sim.read()
            assert 960.0 <= r.pressure_hpa <= 1060.0

    def test_wind_speed_non_negative(self):
        sim = WeatherSimulator()
        for _ in range(50):
            r = sim.read()
            assert r.wind_speed_kph >= 0.0

    def test_deterministic_with_station_id(self):
        """Same station_id seed should produce reproducible sequences."""
        sim1 = WeatherSimulator(station_id="seed-test")
        sim2 = WeatherSimulator(station_id="seed-test")
        # Same station_id gives same Random seed, so same _rng sequence
        r1 = sim1.read(sim_time=5000.0)
        r2 = sim2.read(sim_time=5000.0)
        assert r1.temperature_c == r2.temperature_c
        assert r1.humidity_pct == r2.humidity_pct


# ---------------------------------------------------------------------------
# WeatherStationAddon tests
# ---------------------------------------------------------------------------

class TestWeatherStationAddon:
    """Tests for the main addon class — lifecycle, gather, services."""

    def test_addon_info(self):
        addon = WeatherStationAddon()
        assert addon.info.id == "weather-station"
        assert addon.info.name == "Weather Station"
        assert addon.info.version == "1.0.0"
        assert addon.info.category == "sensors"

    @pytest.mark.asyncio
    async def test_register_creates_simulator(self):
        addon = WeatherStationAddon()
        await addon.register()
        assert addon._simulator is not None
        assert addon._registered is True
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_unregister_cleans_up(self):
        addon = WeatherStationAddon()
        await addon.register()
        await addon.unregister()
        assert addon._simulator is None
        assert addon._registered is False
        assert addon._readings == []

    @pytest.mark.asyncio
    async def test_register_with_context(self):
        tracker = FakeTargetTracker()
        bus = FakeEventBus()
        addon_bus = AddonEventBus()
        context = AddonContext(
            target_tracker=tracker,
            event_bus=bus,
            addon_event_bus=addon_bus,
            site_id="test-site",
        )
        addon = WeatherStationAddon()
        await addon.register(context=context)

        assert addon.target_tracker is tracker
        assert addon.event_bus is bus
        assert addon.site_id == "test-site"
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_register_with_config_overrides(self):
        context = AddonContext(
            state={
                "weather_station_config": {
                    "poll_interval": 30.0,
                    "max_history": 50,
                    "station_id": "wx-custom",
                }
            }
        )
        addon = WeatherStationAddon()
        await addon.register(context=context)

        assert addon._poll_interval == 30.0
        assert addon._max_history == 50
        assert addon.station_id == "wx-custom"
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_gather_returns_target(self):
        addon = WeatherStationAddon(station_id="gather-test")
        await addon.register()

        targets = await addon.gather()
        assert len(targets) == 1
        t = targets[0]
        assert t["target_id"] == "wx_gather-test"
        assert t["source"] == "weather"
        assert "position" in t
        assert "lat" in t["position"]
        assert "lng" in t["position"]
        assert t["classification"] == "weather_station"
        assert "conditions" in t
        assert "severity" in t
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_gather_stores_reading(self):
        addon = WeatherStationAddon()
        await addon.register()

        assert len(addon.readings) == 0
        await addon.gather()
        assert len(addon.readings) == 1
        await addon.gather()
        assert len(addon.readings) == 2
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_gather_publishes_addon_event(self):
        addon_bus = AddonEventBus()
        events_received: list = []
        addon_bus.subscribe("addon:weather-station:*", lambda e: events_received.append(e))

        context = AddonContext(addon_event_bus=addon_bus)
        addon = WeatherStationAddon()
        await addon.register(context=context)

        await addon.gather()
        # Should have at least a weather_reading event
        reading_events = [e for e in events_received if e.event_type == "weather_reading"]
        assert len(reading_events) == 1
        assert "temperature_c" in reading_events[0].data
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_gather_updates_target_tracker(self):
        tracker = FakeTargetTracker()
        context = AddonContext(target_tracker=tracker)
        addon = WeatherStationAddon(station_id="track-test")
        await addon.register(context=context)

        await addon.gather()
        assert "wx_track-test" in tracker.targets
        target = tracker.targets["wx_track-test"]
        assert target["source"] == "weather"
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_gather_publishes_to_event_bus(self):
        bus = FakeEventBus()
        context = AddonContext(event_bus=bus)
        addon = WeatherStationAddon()
        await addon.register(context=context)

        await addon.gather()
        assert len(bus.events) == 1
        topic, data, source = bus.events[0]
        assert topic == "sensor:weather:reading"
        assert source == "weather-station"
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_gather_returns_empty_before_register(self):
        addon = WeatherStationAddon()
        # gather() without register() — simulator is None
        targets = await addon.gather()
        assert targets == []

    @pytest.mark.asyncio
    async def test_max_history_enforced(self):
        addon = WeatherStationAddon()
        addon._max_history = 5
        await addon.register()

        for _ in range(10):
            await addon.gather()
        assert len(addon.readings) == 5
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_latest_reading_property(self):
        addon = WeatherStationAddon()
        assert addon.latest_reading is None

        await addon.register()
        await addon.gather()
        assert addon.latest_reading is not None
        assert isinstance(addon.latest_reading, WeatherReading)
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_get_summary_no_data(self):
        addon = WeatherStationAddon()
        summary = addon.get_summary()
        assert summary["status"] == "no_data"

    @pytest.mark.asyncio
    async def test_get_summary_with_data(self):
        addon = WeatherStationAddon(station_id="sum-test")
        await addon.register()
        for _ in range(5):
            await addon.gather()

        summary = addon.get_summary()
        assert summary["station_id"] == "sum-test"
        assert summary["reading_count"] == 5
        assert "latest" in summary
        assert "conditions" in summary
        assert "avg_temp_c" in summary
        assert "avg_humidity_pct" in summary
        assert "avg_wind_kph" in summary
        assert "max_wind_kph" in summary
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_to_geojson_empty(self):
        addon = WeatherStationAddon()
        geojson = addon.to_geojson()
        assert geojson["type"] == "FeatureCollection"
        assert geojson["features"] == []

    @pytest.mark.asyncio
    async def test_to_geojson_with_reading(self):
        addon = WeatherStationAddon(station_id="geo-test", lat=40.0, lng=-105.0)
        await addon.register()
        await addon.gather()

        geojson = addon.to_geojson()
        assert len(geojson["features"]) == 1
        feat = geojson["features"][0]
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "Point"
        assert feat["geometry"]["coordinates"] == [-105.0, 40.0]
        assert feat["properties"]["station_id"] == "geo-test"
        await addon.unregister()

    def test_get_panels(self):
        addon = WeatherStationAddon()
        panels = addon.get_panels()
        assert len(panels) == 2
        ids = [p["id"] for p in panels]
        assert "weather-current" in ids
        assert "weather-history" in ids

    def test_get_layers(self):
        addon = WeatherStationAddon()
        layers = addon.get_layers()
        assert len(layers) == 1
        assert layers[0]["id"] == "weather-conditions"

    def test_get_geojson_layers(self):
        addon = WeatherStationAddon()
        glayers = addon.get_geojson_layers()
        assert len(glayers) == 1
        gl = glayers[0]
        assert gl.layer_id == "weather-stations"
        assert gl.addon_id == "weather-station"
        assert gl.refresh_interval == 30
        assert gl.visible_by_default is True

    def test_get_shortcuts(self):
        addon = WeatherStationAddon()
        shortcuts = addon.get_shortcuts()
        assert len(shortcuts) == 1
        assert shortcuts[0]["key"] == "Shift+W"

    def test_get_context_menu_items(self):
        addon = WeatherStationAddon()
        items = addon.get_context_menu_items()
        assert len(items) == 1
        assert items[0]["action"] == "view_weather_at_point"

    @pytest.mark.asyncio
    async def test_health_check_before_register(self):
        addon = WeatherStationAddon()
        h = addon.health_check()
        assert h["status"] == "not_registered"

    @pytest.mark.asyncio
    async def test_health_check_after_register(self):
        addon = WeatherStationAddon(station_id="hc-test")
        await addon.register()
        h = addon.health_check()
        assert h["status"] == "ok"
        assert h["station_id"] == "hc-test"
        assert h["reading_count"] == 0
        assert h["detail"] == "No readings yet"
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_health_check_with_readings(self):
        addon = WeatherStationAddon()
        await addon.register()
        await addon.gather()
        h = addon.health_check()
        assert h["reading_count"] == 1
        assert "last_reading_age_s" in h
        await addon.unregister()

    @pytest.mark.asyncio
    async def test_alert_on_extreme_heat(self):
        addon_bus = AddonEventBus()
        alerts: list = []
        addon_bus.subscribe(
            "addon:weather-station:weather_alert",
            lambda e: alerts.append(e),
        )
        context = AddonContext(addon_event_bus=addon_bus)
        addon = WeatherStationAddon()
        await addon.register(context=context)

        # Force a hot reading
        addon._simulator._rng = type(addon._simulator._rng)(42)
        addon._simulator.base_temp_c = 45.0
        addon._simulator.temp_amplitude_c = 0.0

        await addon.gather()
        # Check if extreme_heat alert was published
        heat_alerts = [a for a in alerts if a.data.get("type") == "extreme_heat"]
        assert len(heat_alerts) >= 1
        assert heat_alerts[0].data["severity"] == 2
        await addon.unregister()

    def test_repr(self):
        addon = WeatherStationAddon()
        r = repr(addon)
        assert "WeatherStationAddon" in r
        assert "weather-station" in r


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------

class TestWeatherStationManifest:
    """Tests for the TOML manifest file."""

    def test_load_manifest(self):
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "tritium_lib"
            / "sdk"
            / "examples"
            / "weather_station"
            / "tritium_addon.toml"
        )
        m = load_manifest(manifest_path)
        assert m.id == "weather-station"
        assert m.name == "Weather Station"
        assert m.version == "1.0.0"

    def test_manifest_validates(self):
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "tritium_lib"
            / "sdk"
            / "examples"
            / "weather_station"
            / "tritium_addon.toml"
        )
        m = load_manifest(manifest_path)
        errors = validate_manifest(m)
        assert errors == [], f"Manifest validation errors: {errors}"

    def test_manifest_category(self):
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "tritium_lib"
            / "sdk"
            / "examples"
            / "weather_station"
            / "tritium_addon.toml"
        )
        m = load_manifest(manifest_path)
        assert m.category_window == "sensors"
        assert m.category_tab_order == 20
        assert m.category_icon == "cloud"

    def test_manifest_permissions(self):
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "tritium_lib"
            / "sdk"
            / "examples"
            / "weather_station"
            / "tritium_addon.toml"
        )
        m = load_manifest(manifest_path)
        assert m.perm_serial is False
        assert m.perm_network is False
        assert m.perm_mqtt is True
        assert m.perm_storage is True

    def test_manifest_panels(self):
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "tritium_lib"
            / "sdk"
            / "examples"
            / "weather_station"
            / "tritium_addon.toml"
        )
        m = load_manifest(manifest_path)
        assert len(m.panels) == 2
        panel_ids = [p["id"] for p in m.panels]
        assert "weather-current" in panel_ids
        assert "weather-history" in panel_ids

    def test_manifest_config_fields(self):
        manifest_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "tritium_lib"
            / "sdk"
            / "examples"
            / "weather_station"
            / "tritium_addon.toml"
        )
        m = load_manifest(manifest_path)
        assert "poll_interval" in m.config_fields
        assert m.config_fields["poll_interval"]["default"] == 10.0


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------

class TestWeatherStationRunner:
    """Tests for the headless runner."""

    @pytest.mark.asyncio
    async def test_discover_devices(self):
        from tritium_lib.sdk.examples.weather_station.runner import WeatherStationRunner

        runner = WeatherStationRunner(station_id="run-test")
        devices = await runner.discover_devices()
        assert len(devices) == 1
        assert devices[0]["id"] == "run-test"
        assert devices[0]["type"] == "weather_station_sim"

    @pytest.mark.asyncio
    async def test_start_and_stop_device(self):
        from tritium_lib.sdk.examples.weather_station.runner import WeatherStationRunner

        runner = WeatherStationRunner(station_id="run-test", poll_interval=100.0)
        ok = await runner.start_device({"id": "run-test", "lat": 40.0, "lng": -105.0})
        assert ok is True
        assert runner._simulator is not None
        assert runner._poll_task is not None

        stopped = await runner.stop_device("run-test")
        assert stopped is True
        assert runner._simulator is None

    @pytest.mark.asyncio
    async def test_on_command_get_reading(self):
        from tritium_lib.sdk.examples.weather_station.runner import WeatherStationRunner

        runner = WeatherStationRunner(station_id="cmd-test")
        await runner.start_device({"id": "cmd-test"})

        result = await runner.on_command("get_reading", {})
        assert "temperature_c" in result
        assert "station_id" in result
        await runner.stop_device("cmd-test")

    @pytest.mark.asyncio
    async def test_on_command_get_reading_before_start(self):
        from tritium_lib.sdk.examples.weather_station.runner import WeatherStationRunner

        runner = WeatherStationRunner(station_id="cmd-test")
        result = await runner.on_command("get_reading", {})
        assert result == {"error": "not_started"}

    @pytest.mark.asyncio
    async def test_on_command_set_interval(self):
        from tritium_lib.sdk.examples.weather_station.runner import WeatherStationRunner

        runner = WeatherStationRunner(station_id="int-test", poll_interval=10.0)
        result = await runner.on_command("set_interval", {"interval": 30.0})
        assert result["interval"] == 30.0
        assert runner.poll_interval == 30.0

    @pytest.mark.asyncio
    async def test_on_command_get_status(self):
        from tritium_lib.sdk.examples.weather_station.runner import WeatherStationRunner

        runner = WeatherStationRunner(station_id="stat-test")
        result = await runner.on_command("get_status", {})
        assert result["station_id"] == "stat-test"
        assert result["reading_count"] == 0

    @pytest.mark.asyncio
    async def test_on_command_unknown(self):
        from tritium_lib.sdk.examples.weather_station.runner import WeatherStationRunner

        runner = WeatherStationRunner(station_id="unk-test")
        result = await runner.on_command("explode", {})
        assert "error" in result
        assert "unknown_command" in result["error"]

    def test_topic_helpers(self):
        from tritium_lib.sdk.examples.weather_station.runner import WeatherStationRunner

        runner = WeatherStationRunner(
            station_id="topic-test", site_id="campus"
        )
        assert runner.status_topic == "tritium/campus/weather/topic-test/status"
        assert runner.command_topic == "tritium/campus/weather/topic-test/command"
        assert runner.data_topic("reading") == "tritium/campus/weather/topic-test/reading"
        assert runner.data_topic("reading", "wx-2") == "tritium/campus/weather/wx-2/reading"

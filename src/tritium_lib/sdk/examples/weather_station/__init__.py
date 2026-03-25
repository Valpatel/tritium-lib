# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under Apache-2.0 — see LICENSE for details.
"""Weather Station — example addon demonstrating the full Tritium SDK lifecycle.

This addon simulates a weather station sensor that reads temperature,
humidity, wind speed, and barometric pressure.  It publishes weather
events via the AddonEventBus, registers weather-condition targets with
the TargetTracker, and exposes a GeoJSON map layer.

Use this as a template when building new addons.

Usage::

    from tritium_lib.sdk.examples.weather_station import WeatherStationAddon

    addon = WeatherStationAddon()
    await addon.register(context=my_context)
    readings = await addon.gather()
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ...addon_base import AddonBase, AddonInfo
from ...addon_events import AddonEventBus
from ...config_loader import AddonConfig
from ...context import AddonContext
from ...geo_layer import AddonGeoLayer
from ...interfaces import SensorAddon
from ...protocols import IEventBus, ITargetTracker


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class WeatherReading:
    """A single weather observation from a station."""

    station_id: str
    timestamp: float = field(default_factory=time.time)
    temperature_c: float = 20.0
    humidity_pct: float = 50.0
    wind_speed_kph: float = 0.0
    wind_direction_deg: float = 0.0
    pressure_hpa: float = 1013.25
    rain_mm_hr: float = 0.0
    visibility_km: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            "station_id": self.station_id,
            "timestamp": self.timestamp,
            "temperature_c": round(self.temperature_c, 1),
            "humidity_pct": round(self.humidity_pct, 1),
            "wind_speed_kph": round(self.wind_speed_kph, 1),
            "wind_direction_deg": round(self.wind_direction_deg, 1),
            "pressure_hpa": round(self.pressure_hpa, 2),
            "rain_mm_hr": round(self.rain_mm_hr, 1),
            "visibility_km": round(self.visibility_km, 1),
        }

    @property
    def temperature_f(self) -> float:
        """Temperature in Fahrenheit."""
        return self.temperature_c * 9.0 / 5.0 + 32.0

    @property
    def wind_chill_c(self) -> float:
        """Wind chill in Celsius (valid when temp <= 10C, wind >= 4.8 kph)."""
        t = self.temperature_c
        v = self.wind_speed_kph
        if t > 10.0 or v < 4.8:
            return t
        return (
            13.12
            + 0.6215 * t
            - 11.37 * (v ** 0.16)
            + 0.3965 * t * (v ** 0.16)
        )

    @property
    def heat_index_c(self) -> float:
        """Simplified heat index in Celsius (valid when temp >= 27C)."""
        t = self.temperature_c
        rh = self.humidity_pct
        if t < 27.0:
            return t
        # Rothfusz regression
        tf = self.temperature_f
        hi_f = (
            -42.379
            + 2.04901523 * tf
            + 10.14333127 * rh
            - 0.22475541 * tf * rh
            - 6.83783e-3 * tf ** 2
            - 5.481717e-2 * rh ** 2
            + 1.22874e-3 * tf ** 2 * rh
            + 8.5282e-4 * tf * rh ** 2
            - 1.99e-6 * tf ** 2 * rh ** 2
        )
        return (hi_f - 32.0) * 5.0 / 9.0

    def classify_conditions(self) -> str:
        """Return a human-readable condition string."""
        if self.rain_mm_hr > 7.5:
            return "heavy_rain"
        if self.rain_mm_hr > 2.5:
            return "moderate_rain"
        if self.rain_mm_hr > 0.1:
            return "light_rain"
        if self.visibility_km < 1.0:
            return "fog"
        if self.wind_speed_kph > 60.0:
            return "storm"
        if self.wind_speed_kph > 30.0:
            return "windy"
        if self.temperature_c > 35.0:
            return "extreme_heat"
        if self.temperature_c < -10.0:
            return "extreme_cold"
        if self.humidity_pct > 90.0:
            return "humid"
        return "clear"

    def severity_level(self) -> int:
        """Return 0-3 severity (0 = normal, 3 = severe)."""
        cond = self.classify_conditions()
        severity_map = {
            "clear": 0,
            "humid": 0,
            "light_rain": 1,
            "moderate_rain": 1,
            "windy": 1,
            "fog": 1,
            "heavy_rain": 2,
            "extreme_heat": 2,
            "extreme_cold": 2,
            "storm": 3,
        }
        return severity_map.get(cond, 0)


# ---------------------------------------------------------------------------
# Weather simulator
# ---------------------------------------------------------------------------

class WeatherSimulator:
    """Generates realistic simulated weather readings using sinusoidal
    day/night temperature cycles with random perturbation.
    """

    def __init__(
        self,
        station_id: str = "wx-sim-001",
        lat: float = 39.7392,
        lng: float = -104.9903,
        base_temp_c: float = 20.0,
        temp_amplitude_c: float = 8.0,
    ) -> None:
        self.station_id = station_id
        self.lat = lat
        self.lng = lng
        self.base_temp_c = base_temp_c
        self.temp_amplitude_c = temp_amplitude_c
        self._rng = random.Random(hash(station_id))
        self._time_offset = self._rng.uniform(0.0, 2 * math.pi)
        self._pressure_trend: float = 0.0  # hPa/hour drift
        self._last_pressure: float = 1013.25

    def read(self, sim_time: float | None = None) -> WeatherReading:
        """Generate a weather reading for the given timestamp (or now)."""
        t = sim_time if sim_time is not None else time.time()

        # Day/night cycle: period = 86400s
        day_phase = (t / 86400.0) * 2 * math.pi + self._time_offset
        temp = self.base_temp_c + self.temp_amplitude_c * math.sin(day_phase)
        temp += self._rng.gauss(0, 1.5)  # random noise

        # Humidity inversely correlated with temperature
        humidity = max(10.0, min(100.0, 70.0 - (temp - 20.0) * 1.5 + self._rng.gauss(0, 5)))

        # Pressure random walk
        self._pressure_trend += self._rng.gauss(0, 0.1)
        self._pressure_trend = max(-2.0, min(2.0, self._pressure_trend))
        self._last_pressure += self._pressure_trend * 0.01
        self._last_pressure = max(960.0, min(1060.0, self._last_pressure))

        # Wind
        wind_speed = max(0.0, self._rng.gauss(12.0, 8.0))
        wind_dir = self._rng.uniform(0.0, 360.0)

        # Rain (probability-based)
        rain = 0.0
        if humidity > 80.0 and self._rng.random() < 0.3:
            rain = self._rng.expovariate(0.2)  # mean = 5 mm/hr

        # Visibility
        visibility = 10.0
        if rain > 5.0:
            visibility = self._rng.uniform(0.5, 3.0)
        elif humidity > 95.0:
            visibility = self._rng.uniform(0.2, 2.0)

        return WeatherReading(
            station_id=self.station_id,
            timestamp=t,
            temperature_c=temp,
            humidity_pct=humidity,
            wind_speed_kph=wind_speed,
            wind_direction_deg=wind_dir,
            pressure_hpa=self._last_pressure,
            rain_mm_hr=rain,
            visibility_km=visibility,
        )


# ---------------------------------------------------------------------------
# Main addon class
# ---------------------------------------------------------------------------

class WeatherStationAddon(SensorAddon):
    """Example sensor addon: simulated weather station.

    Demonstrates:
    - Extending SensorAddon with a gather() implementation
    - Publishing events via AddonEventBus
    - Registering targets with ITargetTracker
    - Providing GeoJSON map layers
    - Providing panels, layers, and keyboard shortcuts
    - Health checks
    - Configuration via AddonConfig
    """

    info = AddonInfo(
        id="weather-station",
        name="Weather Station",
        version="1.0.0",
        description="Simulated weather station sensor addon",
        author="Tritium SDK Team",
        license="Apache-2.0",
        category="sensors",
        icon="cloud",
        min_sdk_version="1.0.0",
    )

    def __init__(
        self,
        station_id: str = "wx-sim-001",
        lat: float = 39.7392,
        lng: float = -104.9903,
    ) -> None:
        super().__init__()
        self.station_id = station_id
        self.lat = lat
        self.lng = lng
        self._simulator: WeatherSimulator | None = None
        self._config: AddonConfig | None = None
        self._readings: list[WeatherReading] = []
        self._max_history: int = 100
        self._poll_task: asyncio.Task | None = None
        self._poll_interval: float = 10.0  # seconds
        self._alert_thresholds: dict[str, float] = {
            "temp_high_c": 40.0,
            "temp_low_c": -15.0,
            "wind_high_kph": 80.0,
            "rain_heavy_mm_hr": 10.0,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def register(
        self, app: Any = None, *, context: AddonContext | None = None
    ) -> None:
        """Initialize the weather simulator and wire up services."""
        await super().register(app, context=context)

        # Load config overrides from context state if available
        if context and context.state:
            config_data = context.state.get("weather_station_config", {})
            self._config = AddonConfig(
                config_schema={
                    "poll_interval": {"type": "float", "default": 10.0},
                    "max_history": {"type": "int", "default": 100},
                    "station_id": {"type": "str", "default": "wx-sim-001"},
                    "lat": {"type": "float", "default": 39.7392},
                    "lng": {"type": "float", "default": -104.9903},
                },
                overrides=config_data,
            )
            if self._config.get("poll_interval"):
                self._poll_interval = float(self._config.get("poll_interval"))
            if self._config.get("max_history"):
                self._max_history = int(self._config.get("max_history"))
            sid = self._config.get("station_id")
            if sid:
                self.station_id = sid
            lat_cfg = self._config.get("lat")
            if lat_cfg is not None:
                self.lat = float(lat_cfg)
            lng_cfg = self._config.get("lng")
            if lng_cfg is not None:
                self.lng = float(lng_cfg)

        # Create simulator
        self._simulator = WeatherSimulator(
            station_id=self.station_id,
            lat=self.lat,
            lng=self.lng,
        )

    async def unregister(self, app: Any = None) -> None:
        """Clean up: cancel poll task, clear readings."""
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._poll_task = None

        self._readings.clear()
        self._simulator = None
        await super().unregister(app)

    # ------------------------------------------------------------------
    # SensorAddon.gather()
    # ------------------------------------------------------------------

    async def gather(self) -> list[dict]:
        """Read weather data and return target dicts.

        This is the core SensorAddon method.  Each call:
        1. Reads a new weather observation from the simulator
        2. Stores it in the history ring buffer
        3. Publishes a weather_reading event via AddonEventBus
        4. Checks alert thresholds and publishes weather_alert if exceeded
        5. Updates the target tracker with a weather-condition target
        6. Returns the target dict for pipeline consumption
        """
        if self._simulator is None:
            return []

        reading = self._simulator.read()
        self._store_reading(reading)

        # Publish reading event
        self.publish_addon_event(
            event_type="weather_reading",
            data=reading.to_dict(),
            device_id=self.station_id,
        )

        # Check alert thresholds
        alerts = self._check_alerts(reading)
        for alert in alerts:
            self.publish_addon_event(
                event_type="weather_alert",
                data=alert,
                device_id=self.station_id,
            )

        # Build a target for the tracker
        target = self._reading_to_target(reading)

        # Register with target tracker if available
        if self.target_tracker is not None:
            self.target_tracker.update_target(target["target_id"], target)

        # Publish to event bus if available
        if self.event_bus is not None:
            self.event_bus.publish(
                "sensor:weather:reading",
                data=reading.to_dict(),
                source=self.info.id,
            )

        return [target]

    # ------------------------------------------------------------------
    # Polling loop (for use when running inside SC or standalone)
    # ------------------------------------------------------------------

    async def start_polling(self) -> None:
        """Start a background task that calls gather() periodically."""
        if self._poll_task is not None:
            return
        self._poll_task = asyncio.ensure_future(self._poll_loop())
        self._background_tasks.append(self._poll_task)

    async def _poll_loop(self) -> None:
        """Internal polling coroutine."""
        while True:
            try:
                await self.gather()
            except Exception:
                pass  # Swallow errors to keep polling
            await asyncio.sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # Panels, layers, shortcuts
    # ------------------------------------------------------------------

    def get_panels(self) -> list[dict]:
        return [
            {
                "id": "weather-current",
                "title": "Weather Station",
                "file": "weather-panel.js",
                "category": "sensors",
                "tab_order": 20,
            },
            {
                "id": "weather-history",
                "title": "Weather History",
                "file": "weather-history.js",
                "category": "sensors",
                "tab_order": 21,
            },
        ]

    def get_layers(self) -> list[dict]:
        return [
            {
                "id": "weather-conditions",
                "label": "Weather Conditions",
                "category": "Environment",
                "color": "#00bfff",
                "key": "showWeatherConditions",
            }
        ]

    def get_geojson_layers(self) -> list[AddonGeoLayer]:
        return [
            AddonGeoLayer(
                layer_id="weather-stations",
                addon_id=self.info.id,
                label="Weather Stations",
                category="Environment",
                color="#00bfff",
                geojson_endpoint="/api/addons/weather-station/geojson",
                refresh_interval=30,
                visible_by_default=True,
            )
        ]

    def get_shortcuts(self) -> list[dict]:
        return [
            {
                "key": "Shift+W",
                "action": "toggle_weather_layer",
                "description": "Toggle weather conditions layer",
            }
        ]

    def get_context_menu_items(self) -> list[dict]:
        return [
            {
                "label": "View Weather Here",
                "action": "view_weather_at_point",
                "when": "always",
            }
        ]

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        """Report addon health including reading count and last reading age."""
        base = super().health_check()
        base["station_id"] = self.station_id
        base["reading_count"] = len(self._readings)

        if self._readings:
            age = time.time() - self._readings[-1].timestamp
            base["last_reading_age_s"] = round(age, 1)
            if age > self._poll_interval * 3:
                base["status"] = "degraded"
                base["detail"] = "Readings are stale"
        else:
            base["detail"] = "No readings yet"

        return base

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    @property
    def latest_reading(self) -> WeatherReading | None:
        """The most recent weather reading, or None."""
        return self._readings[-1] if self._readings else None

    @property
    def readings(self) -> list[WeatherReading]:
        """Full reading history (newest last)."""
        return list(self._readings)

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of recent weather conditions."""
        if not self._readings:
            return {"station_id": self.station_id, "status": "no_data"}

        recent = self._readings[-10:]  # last 10 readings
        return {
            "station_id": self.station_id,
            "reading_count": len(self._readings),
            "latest": self._readings[-1].to_dict(),
            "conditions": self._readings[-1].classify_conditions(),
            "severity": self._readings[-1].severity_level(),
            "avg_temp_c": round(
                sum(r.temperature_c for r in recent) / len(recent), 1
            ),
            "avg_humidity_pct": round(
                sum(r.humidity_pct for r in recent) / len(recent), 1
            ),
            "avg_wind_kph": round(
                sum(r.wind_speed_kph for r in recent) / len(recent), 1
            ),
            "max_wind_kph": round(max(r.wind_speed_kph for r in recent), 1),
            "total_rain_mm": round(
                sum(r.rain_mm_hr for r in recent) * (self._poll_interval / 3600.0),
                2,
            ),
        }

    def to_geojson(self) -> dict:
        """Return a GeoJSON FeatureCollection for the station."""
        features = []
        if self._readings:
            r = self._readings[-1]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [self.lng, self.lat],
                },
                "properties": {
                    "station_id": self.station_id,
                    "temperature_c": r.temperature_c,
                    "humidity_pct": r.humidity_pct,
                    "wind_speed_kph": r.wind_speed_kph,
                    "conditions": r.classify_conditions(),
                    "severity": r.severity_level(),
                },
            })
        return {"type": "FeatureCollection", "features": features}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_reading(self, reading: WeatherReading) -> None:
        """Add a reading to the ring buffer."""
        self._readings.append(reading)
        if len(self._readings) > self._max_history:
            self._readings = self._readings[-self._max_history:]

    def _reading_to_target(self, reading: WeatherReading) -> dict:
        """Convert a WeatherReading into a TrackedTarget-compatible dict."""
        conditions = reading.classify_conditions()
        return {
            "target_id": f"wx_{self.station_id}",
            "source": "weather",
            "position": {"lat": self.lat, "lng": self.lng},
            "classification": "weather_station",
            "alliance": "neutral",
            "label": f"WX {self.station_id}",
            "data": reading.to_dict(),
            "conditions": conditions,
            "severity": reading.severity_level(),
            "last_seen": reading.timestamp,
        }

    def _check_alerts(self, reading: WeatherReading) -> list[dict]:
        """Check reading against alert thresholds. Return alert dicts."""
        alerts: list[dict] = []

        if reading.temperature_c > self._alert_thresholds["temp_high_c"]:
            alerts.append({
                "type": "extreme_heat",
                "message": f"Temperature {reading.temperature_c:.1f}C exceeds {self._alert_thresholds['temp_high_c']}C",
                "value": reading.temperature_c,
                "threshold": self._alert_thresholds["temp_high_c"],
                "severity": 2,
            })

        if reading.temperature_c < self._alert_thresholds["temp_low_c"]:
            alerts.append({
                "type": "extreme_cold",
                "message": f"Temperature {reading.temperature_c:.1f}C below {self._alert_thresholds['temp_low_c']}C",
                "value": reading.temperature_c,
                "threshold": self._alert_thresholds["temp_low_c"],
                "severity": 2,
            })

        if reading.wind_speed_kph > self._alert_thresholds["wind_high_kph"]:
            alerts.append({
                "type": "high_wind",
                "message": f"Wind {reading.wind_speed_kph:.1f} kph exceeds {self._alert_thresholds['wind_high_kph']} kph",
                "value": reading.wind_speed_kph,
                "threshold": self._alert_thresholds["wind_high_kph"],
                "severity": 3,
            })

        if reading.rain_mm_hr > self._alert_thresholds["rain_heavy_mm_hr"]:
            alerts.append({
                "type": "heavy_rain",
                "message": f"Rain {reading.rain_mm_hr:.1f} mm/hr exceeds {self._alert_thresholds['rain_heavy_mm_hr']} mm/hr",
                "value": reading.rain_mm_hr,
                "threshold": self._alert_thresholds["rain_heavy_mm_hr"],
                "severity": 2,
            })

        return alerts


__all__ = [
    "WeatherStationAddon",
    "WeatherReading",
    "WeatherSimulator",
]

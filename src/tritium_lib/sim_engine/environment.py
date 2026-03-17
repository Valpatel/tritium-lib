# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Weather and environment simulation for the Tritium sim engine.

Simulates time-of-day lighting, weather conditions, and their combined
effects on visibility, movement, accuracy, and sound propagation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Time of Day
# ---------------------------------------------------------------------------

class TimeOfDay:
    """Simulates a 24-hour clock with lighting and visibility calculations.

    Hour is a float in [0, 24). Noon (12.0) is brightest, midnight (0.0)
    is darkest.  Twilight bands sit between day and night.
    """

    def __init__(self, hour: float = 12.0) -> None:
        self.hour: float = hour % 24.0

    def advance(self, dt_hours: float) -> None:
        """Advance time, wrapping at 24."""
        self.hour = (self.hour + dt_hours) % 24.0

    # -- predicates --

    def is_day(self) -> bool:
        """True between 06:00 and 18:00."""
        return 6.0 <= self.hour <= 18.0

    def is_night(self) -> bool:
        """True between 21:00 and 05:00 (next day)."""
        return self.hour >= 21.0 or self.hour <= 5.0

    def is_twilight(self) -> bool:
        """True during dawn (5-6) or dusk (18-21) transitions."""
        return (5.0 < self.hour < 6.0) or (18.0 < self.hour < 21.0)

    # -- continuous values --

    def light_level(self) -> float:
        """0.0 at midnight, 1.0 at noon. Smooth sine curve."""
        # Map hour to angle: midnight=0 => sin minimum, noon=12 => sin maximum
        angle = (self.hour - 6.0) * math.pi / 12.0
        raw = (math.sin(angle) + 1.0) / 2.0
        return max(0.0, min(1.0, raw))

    def visibility_modifier(self) -> float:
        """1.0 during full day, 0.3 at deepest night, smooth transitions."""
        ll = self.light_level()
        return 0.3 + 0.7 * ll

    def sun_angle(self) -> float:
        """Sun angle in degrees: 0 at horizon, 90 at zenith.

        Negative values are possible (sun below horizon) but clamped to 0.
        """
        # Peak at solar noon (12:00), -90 at midnight
        angle = 90.0 * math.sin((self.hour - 6.0) * math.pi / 12.0)
        return max(0.0, angle)


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

class Weather(Enum):
    """Discrete weather conditions."""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    FOG = "fog"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    STORM = "storm"
    SANDSTORM = "sandstorm"


@dataclass
class WeatherState:
    """Instantaneous weather parameters."""
    current: Weather = Weather.CLEAR
    intensity: float = 0.0        # 0-1
    wind_speed: float = 0.0       # m/s
    wind_direction: float = 0.0   # radians
    temperature: float = 20.0     # celsius
    humidity: float = 0.5         # 0-1


# ---------------------------------------------------------------------------
# Weather Effects
# ---------------------------------------------------------------------------

class WeatherEffects:
    """Static methods computing gameplay modifiers from weather state."""

    _VISIBILITY: dict[Weather, float] = {
        Weather.CLEAR: 1.0,
        Weather.CLOUDY: 0.9,
        Weather.FOG: 0.2,
        Weather.RAIN: 0.6,
        Weather.HEAVY_RAIN: 0.4,
        Weather.SNOW: 0.5,
        Weather.STORM: 0.3,
        Weather.SANDSTORM: 0.15,
    }

    _MOVEMENT: dict[Weather, float] = {
        Weather.CLEAR: 1.0,
        Weather.CLOUDY: 1.0,
        Weather.FOG: 0.9,
        Weather.RAIN: 0.85,
        Weather.HEAVY_RAIN: 0.8,
        Weather.SNOW: 0.6,
        Weather.STORM: 0.4,
        Weather.SANDSTORM: 0.5,
    }

    @staticmethod
    def visibility_modifier(weather: WeatherState) -> float:
        """Visibility multiplier from weather conditions."""
        base = WeatherEffects._VISIBILITY.get(weather.current, 1.0)
        # Higher intensity further reduces visibility
        return max(0.05, base * (1.0 - 0.3 * weather.intensity))

    @staticmethod
    def movement_modifier(weather: WeatherState) -> float:
        """Movement speed multiplier from weather conditions."""
        base = WeatherEffects._MOVEMENT.get(weather.current, 1.0)
        return max(0.1, base * (1.0 - 0.1 * weather.intensity))

    @staticmethod
    def accuracy_modifier(weather: WeatherState) -> float:
        """Accuracy multiplier. Wind and rain degrade it."""
        wind_penalty = 0.02 * weather.wind_speed
        rain_penalty = 0.0
        if weather.current in (Weather.RAIN, Weather.HEAVY_RAIN, Weather.STORM):
            rain_penalty = 0.2 * weather.intensity
        return max(0.1, 1.0 - wind_penalty - rain_penalty)

    @staticmethod
    def sound_modifier(weather: WeatherState) -> float:
        """Sound detection range multiplier. Heavy weather masks sounds."""
        masking: dict[Weather, float] = {
            Weather.CLEAR: 1.0,
            Weather.CLOUDY: 1.0,
            Weather.FOG: 0.95,
            Weather.RAIN: 0.7,
            Weather.HEAVY_RAIN: 0.5,
            Weather.SNOW: 0.85,
            Weather.STORM: 0.3,
            Weather.SANDSTORM: 0.4,
        }
        base = masking.get(weather.current, 1.0)
        return max(0.1, base * (1.0 - 0.15 * weather.intensity))

    @staticmethod
    def combined_visibility(weather: WeatherState, time: TimeOfDay) -> float:
        """Combined visibility from weather and time-of-day effects."""
        return WeatherEffects.visibility_modifier(weather) * time.visibility_modifier()


# ---------------------------------------------------------------------------
# Weather Simulator
# ---------------------------------------------------------------------------

# Transition probabilities per hour: {from_weather: [(to_weather, prob), ...]}
_TRANSITIONS: dict[Weather, list[tuple[Weather, float]]] = {
    Weather.CLEAR: [
        (Weather.CLOUDY, 0.10),
        (Weather.FOG, 0.03),
    ],
    Weather.CLOUDY: [
        (Weather.CLEAR, 0.20),
        (Weather.RAIN, 0.15),
        (Weather.FOG, 0.05),
        (Weather.SNOW, 0.03),
    ],
    Weather.FOG: [
        (Weather.CLEAR, 0.25),
        (Weather.CLOUDY, 0.15),
    ],
    Weather.RAIN: [
        (Weather.HEAVY_RAIN, 0.10),
        (Weather.CLOUDY, 0.25),
        (Weather.STORM, 0.05),
    ],
    Weather.HEAVY_RAIN: [
        (Weather.RAIN, 0.30),
        (Weather.STORM, 0.10),
    ],
    Weather.SNOW: [
        (Weather.CLOUDY, 0.20),
        (Weather.CLEAR, 0.05),
    ],
    Weather.STORM: [
        (Weather.RAIN, 0.40),
        (Weather.HEAVY_RAIN, 0.20),
    ],
    Weather.SANDSTORM: [
        (Weather.CLEAR, 0.15),
        (Weather.CLOUDY, 0.10),
    ],
}


class WeatherSimulator:
    """Stochastic weather simulation with gradual transitions."""

    def __init__(
        self,
        initial: Weather = Weather.CLEAR,
        seed: int | None = None,
    ) -> None:
        self.state = WeatherState(current=initial)
        self._rng = random.Random(seed)
        self._storm_remaining: float = 0.0  # hours of storm left

    def update(self, dt_hours: float) -> None:
        """Advance weather simulation by *dt_hours*."""
        if dt_hours <= 0:
            return

        # -- weather transitions --
        if self.state.current == Weather.STORM:
            if self._storm_remaining <= 0:
                self._storm_remaining = self._rng.uniform(1.0, 3.0)
            self._storm_remaining -= dt_hours
            if self._storm_remaining <= 0:
                self.state.current = Weather.RAIN
                self._storm_remaining = 0.0
        else:
            transitions = _TRANSITIONS.get(self.state.current, [])
            for target, prob_per_hour in transitions:
                prob = prob_per_hour * dt_hours
                if self._rng.random() < prob:
                    self.state.current = target
                    if target == Weather.STORM:
                        self._storm_remaining = self._rng.uniform(1.0, 3.0)
                    break

        # -- intensity random walk --
        self.state.intensity += self._rng.gauss(0, 0.05 * dt_hours)
        self.state.intensity = max(0.0, min(1.0, self.state.intensity))

        # -- wind random walk --
        self.state.wind_speed += self._rng.gauss(0, 1.0 * dt_hours)
        self.state.wind_speed = max(0.0, min(30.0, self.state.wind_speed))

        self.state.wind_direction += self._rng.gauss(0, 0.1 * dt_hours)
        self.state.wind_direction %= (2 * math.pi)

        # -- temperature: slight diurnal variation is handled by Environment --
        self.state.temperature += self._rng.gauss(0, 0.2 * dt_hours)

        # -- humidity correlates loosely with weather --
        target_humidity = {
            Weather.CLEAR: 0.3,
            Weather.CLOUDY: 0.5,
            Weather.FOG: 0.95,
            Weather.RAIN: 0.8,
            Weather.HEAVY_RAIN: 0.9,
            Weather.SNOW: 0.7,
            Weather.STORM: 0.85,
            Weather.SANDSTORM: 0.15,
        }.get(self.state.current, 0.5)
        self.state.humidity += (target_humidity - self.state.humidity) * 0.1 * dt_hours
        self.state.humidity = max(0.0, min(1.0, self.state.humidity))


# ---------------------------------------------------------------------------
# Environment  (top-level facade)
# ---------------------------------------------------------------------------

_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _wind_compass(radians: float) -> str:
    """Convert wind direction in radians to 16-point compass label."""
    deg = math.degrees(radians) % 360
    idx = int((deg + 11.25) / 22.5) % 16
    return _COMPASS[idx]


class Environment:
    """Combines TimeOfDay and WeatherSimulator into a single facade."""

    def __init__(
        self,
        time: TimeOfDay | None = None,
        weather: WeatherSimulator | None = None,
    ) -> None:
        self.time = time or TimeOfDay()
        self.weather = weather or WeatherSimulator()

    def update(self, dt_seconds: float) -> None:
        """Advance both time and weather by *dt_seconds*."""
        dt_hours = dt_seconds / 3600.0
        self.time.advance(dt_hours)
        self.weather.update(dt_hours)

    # -- combined modifiers --

    def visibility(self) -> float:
        return WeatherEffects.combined_visibility(self.weather.state, self.time)

    def movement_speed_modifier(self) -> float:
        return WeatherEffects.movement_modifier(self.weather.state)

    def accuracy_modifier(self) -> float:
        return WeatherEffects.accuracy_modifier(self.weather.state)

    def detection_range_modifier(self) -> float:
        """Combined visibility and sound modifier for detection range."""
        vis = self.visibility()
        snd = WeatherEffects.sound_modifier(self.weather.state)
        return (vis + snd) / 2.0

    # -- serialisation --

    def snapshot(self) -> dict:
        """JSON-serializable state dictionary."""
        ws = self.weather.state
        return {
            "hour": round(self.time.hour, 2),
            "is_day": self.time.is_day(),
            "is_night": self.time.is_night(),
            "light_level": round(self.time.light_level(), 3),
            "sun_angle": round(self.time.sun_angle(), 1),
            "weather": ws.current.value,
            "intensity": round(ws.intensity, 3),
            "wind_speed": round(ws.wind_speed, 1),
            "wind_direction": round(math.degrees(ws.wind_direction), 1),
            "temperature": round(ws.temperature, 1),
            "humidity": round(ws.humidity, 3),
            "visibility": round(self.visibility(), 3),
            "movement_modifier": round(self.movement_speed_modifier(), 3),
            "accuracy_modifier": round(self.accuracy_modifier(), 3),
            "detection_range_modifier": round(self.detection_range_modifier(), 3),
        }

    def describe(self) -> str:
        """Human-readable one-liner, e.g. 'Clear day, light wind from NW, 22C'."""
        ws = self.weather.state
        weather_label = ws.current.value.replace("_", " ").title()

        if self.time.is_day():
            period = "day"
        elif self.time.is_night():
            period = "night"
        else:
            period = "twilight"

        if ws.wind_speed < 2.0:
            wind_desc = "calm"
        elif ws.wind_speed < 8.0:
            wind_desc = f"light wind from {_wind_compass(ws.wind_direction)}"
        elif ws.wind_speed < 15.0:
            wind_desc = f"moderate wind from {_wind_compass(ws.wind_direction)}"
        else:
            wind_desc = f"strong wind from {_wind_compass(ws.wind_direction)}"

        temp = f"{ws.temperature:.0f}\u00b0C"
        return f"{weather_label} {period}, {wind_desc}, {temp}"

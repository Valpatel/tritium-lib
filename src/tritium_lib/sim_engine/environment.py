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

    Hour is a float in [0, 24). Solar noon is brightest, solar midnight is
    darkest.  Twilight bands sit between day and night.

    Sunrise/sunset default to 6.0/18.0, which reproduces the classic fixed
    12-hour day.  A :class:`SeasonalCycle` can call :meth:`set_daylight` to
    give the day variable length by latitude/date.  For the default
    sunrise=6/sunset=18 the continuous curves (``light_level``, ``sun_angle``)
    are byte-identical to the historical fixed-day formulas.
    """

    def __init__(
        self,
        hour: float = 12.0,
        sunrise: float = 6.0,
        sunset: float = 18.0,
    ) -> None:
        self.hour: float = hour % 24.0
        self.sunrise: float = sunrise
        self.sunset: float = sunset

    def set_daylight(self, sunrise: float, sunset: float) -> None:
        """Set variable daylight window (e.g. from a SeasonalCycle)."""
        self.sunrise = sunrise
        self.sunset = sunset

    @property
    def solar_noon(self) -> float:
        """Midpoint of the daylight window."""
        return (self.sunrise + self.sunset) / 2.0

    def advance(self, dt_hours: float) -> None:
        """Advance time, wrapping at 24."""
        self.hour = (self.hour + dt_hours) % 24.0

    # -- predicates --

    def is_day(self) -> bool:
        """True between sunrise and sunset (default 06:00-18:00)."""
        return self.sunrise <= self.hour <= self.sunset

    def is_night(self) -> bool:
        """True in deep night: after dusk or before dawn.

        Default (sunrise=6/sunset=18) reproduces the old 21:00 / 05:00 bands.
        """
        return self.hour > self.sunset + 3.0 or self.hour < self.sunrise - 1.0

    def is_twilight(self) -> bool:
        """True during dawn [sunrise-1, sunrise) or dusk (sunset, sunset+3].

        Default (sunrise=6/sunset=18) reproduces the old 5-6 / 18-21 bands.
        """
        dawn = (self.sunrise - 1.0) <= self.hour < self.sunrise
        dusk = self.sunset < self.hour <= self.sunset + 3.0
        return dawn or dusk

    # -- continuous values --

    def light_level(self) -> float:
        """0.0 at solar midnight, 1.0 at solar noon.

        Piecewise so the *default* sunrise=6/sunset=18 is byte-identical to
        the old ``(sin((hour-6)*pi/12)+1)/2`` curve across the whole day.
        """
        day_len = self.sunset - self.sunrise
        if day_len <= 0.0:
            # Polar night: perpetual darkness.
            return 0.0
        if self.sunrise <= self.hour <= self.sunset:
            raw = 0.5 + 0.5 * math.sin(math.pi * (self.hour - self.sunrise) / day_len)
        else:
            # Night arc: 0.5 at the sunrise/sunset edges down to 0 at solar
            # midnight, symmetric.  hours since sunset (wrapped past midnight).
            night_len = 24.0 - day_len
            h = self.hour - self.sunset
            if h < 0.0:
                h += 24.0
            raw = 0.5 - 0.5 * math.sin(math.pi * h / night_len)
        return max(0.0, min(1.0, raw))

    def visibility_modifier(self) -> float:
        """1.0 during full day, 0.3 at deepest night, smooth transitions."""
        ll = self.light_level()
        return 0.3 + 0.7 * ll

    def sun_angle(self) -> float:
        """Sun angle in degrees: 0 at horizon, 90 at zenith.

        Peaks at solar noon, 0 at sunrise/sunset, clamped to >=0 at night.
        For the default sunrise=6/sunset=18 this equals the old
        ``90*sin((hour-6)*pi/12)`` (clamped) during daytime.
        """
        day_len = self.sunset - self.sunrise
        if day_len > 0.0 and self.sunrise <= self.hour <= self.sunset:
            return max(0.0, 90.0 * math.sin(math.pi * (self.hour - self.sunrise) / day_len))
        return 0.0


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
# Seasonal Cycle
# ---------------------------------------------------------------------------

class Season(Enum):
    """The four temperate seasons."""
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"


class SeasonalCycle:
    """Deterministic annual cycle: season, daylight, foliage, temperature.

    Pure and RNG-free -- every method is a closed-form function of
    ``day_of_year`` (1..365) and ``latitude`` (degrees, +N / -S).  Suitable as
    a golden-stable driver of :class:`TimeOfDay` daylight and environmental
    foliage/temperature/snow state.

    Day-of-year default 172 ~ June 21 (northern summer solstice).  Southern
    hemisphere (latitude < 0) flips the calendar by half a year.

    Formulas settled on (see individual methods):
      * ``foliage_state = green_floor + amp*cos(phase)*0.9`` where
        ``amp = 0.45*min(1, |lat|/45)`` and
        ``green_floor = 0.5 + 0.4*(1 - min(1, |lat|/45))``.  Phase is 0 at the
        (hemisphere-corrected) summer solstice, so temperate latitudes swing
        winter-low / summer-high while the tropics stay near-constant evergreen.
      * ``temperature_baseline = annual_mean + seasonal_amp*cos(phase)`` where
        ``annual_mean = 27 - 0.6*|lat|`` and ``seasonal_amp = 3 + 0.35*|lat|``,
        phase 0 at the warmest day (~late July / day 201, hemisphere-corrected).
    """

    def __init__(self, day_of_year: int = 172, latitude: float = 40.0) -> None:
        # Wrap/clamp day into 1..365.
        self.day_of_year: int = ((int(day_of_year) - 1) % 365) + 1
        self.latitude: float = float(latitude)

    # -- helpers --

    def _eff_day(self) -> int:
        """Hemisphere-corrected day-of-year (southern flips by 182)."""
        if self.latitude >= 0.0:
            return self.day_of_year
        return ((self.day_of_year + 182 - 1) % 365) + 1

    # -- season --

    def season(self) -> Season:
        """Season by northern-hemisphere day-of-year bands (southern flips)."""
        d = self._eff_day()
        if d >= 355 or d <= 79:
            return Season.WINTER
        if d <= 171:
            return Season.SPRING
        if d <= 264:
            return Season.SUMMER
        return Season.AUTUMN

    # -- sun geometry --

    def solar_declination_deg(self) -> float:
        """Solar declination in degrees (Cooper's approximation)."""
        return 23.44 * math.sin(math.radians(360.0 * (self.day_of_year - 81) / 365.0))

    def daylight_hours(self) -> float:
        """Length of the day in hours from the sunrise hour-angle formula."""
        decl = math.radians(self.solar_declination_deg())
        lat = math.radians(self.latitude)
        x = -math.tan(lat) * math.tan(decl)
        if x <= -1.0:
            return 24.0  # midnight sun
        if x >= 1.0:
            return 0.0   # polar night
        H = math.degrees(math.acos(x))
        return 2.0 * H / 15.0

    def sunrise_hour(self) -> float:
        """Local sunrise hour, clamped to [0, 12]."""
        return max(0.0, min(12.0, 12.0 - self.daylight_hours() / 2.0))

    def sunset_hour(self) -> float:
        """Local sunset hour, clamped to [12, 24]."""
        return max(12.0, min(24.0, 12.0 + self.daylight_hours() / 2.0))

    # -- foliage / temperature --

    def foliage_state(self) -> float:
        """Green-cover fraction in [0.05, 1.0], hemisphere-aware.

        Temperate latitudes swing from bare winter to lush summer; the tropics
        stay near-constant evergreen.  Phase 0 at the summer solstice.
        """
        eff = self._eff_day()
        phase = 2.0 * math.pi * (eff - 172) / 365.0
        lat_frac = min(1.0, abs(self.latitude) / 45.0)
        amp = 0.45 * lat_frac
        green_floor = 0.5 + 0.4 * (1.0 - lat_frac)
        foliage = green_floor + amp * math.cos(phase) * 0.9
        return max(0.05, min(1.0, foliage))

    def temperature_baseline(self) -> float:
        """Seasonal mean surface temperature in Celsius."""
        annual_mean = 27.0 - 0.6 * abs(self.latitude)
        seasonal_amp = 3.0 + 0.35 * abs(self.latitude)
        eff = self._eff_day()
        phase = 2.0 * math.pi * (eff - 201) / 365.0
        return annual_mean + seasonal_amp * math.cos(phase)

    def precip_is_snow(self, temperature_c: float | None = None) -> bool:
        """True if precipitation would fall as snow at *temperature_c*.

        Falls back to the seasonal baseline temperature when unspecified.
        """
        t = temperature_c if temperature_c is not None else self.temperature_baseline()
        return t <= 1.0

    def weather_bias(self) -> dict[Weather, float]:
        """Climatology multipliers per weather type for the current season.

        Weathers not listed default to 1.0 at the call site.
        """
        s = self.season()
        if s is Season.WINTER:
            return {Weather.SNOW: 3.0, Weather.CLOUDY: 1.5, Weather.FOG: 1.3,
                    Weather.CLEAR: 0.7, Weather.STORM: 0.5}
        if s is Season.SPRING:
            return {Weather.RAIN: 1.6, Weather.CLOUDY: 1.3, Weather.CLEAR: 1.0,
                    Weather.FOG: 1.1}
        if s is Season.SUMMER:
            return {Weather.CLEAR: 1.6, Weather.STORM: 1.4, Weather.RAIN: 0.8,
                    Weather.SNOW: 0.0}
        # AUTUMN
        return {Weather.FOG: 1.6, Weather.RAIN: 1.4, Weather.CLOUDY: 1.3,
                Weather.CLEAR: 0.9}

    # -- serialisation --

    def describe(self) -> str:
        """Human-readable one-liner, e.g. 'Summer (foliage 0.90, ~14.8h daylight)'."""
        return (
            f"{self.season().value.capitalize()} "
            f"(foliage {self.foliage_state():.2f}, "
            f"~{self.daylight_hours():.1f}h daylight)"
        )

    def snapshot(self) -> dict:
        """JSON-serializable seasonal state dictionary."""
        return {
            "season": self.season().value,
            "day_of_year": self.day_of_year,
            "latitude": self.latitude,
            "daylight_hours": round(self.daylight_hours(), 2),
            "sunrise": round(self.sunrise_hour(), 2),
            "sunset": round(self.sunset_hour(), 2),
            "foliage_state": round(self.foliage_state(), 3),
            "temperature_baseline": round(self.temperature_baseline(), 1),
            "is_snow_season": self.precip_is_snow(),
        }


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
        seasonal: SeasonalCycle | None = None,
    ) -> None:
        self.time = time or TimeOfDay()
        self.weather = weather or WeatherSimulator()
        self.seasonal = seasonal
        self.snow_depth_cm: float = 0.0
        if seasonal is not None:
            self.time.set_daylight(seasonal.sunrise_hour(), seasonal.sunset_hour())

    def update(self, dt_seconds: float) -> None:
        """Advance both time and weather by *dt_seconds*."""
        dt_hours = dt_seconds / 3600.0
        self.time.advance(dt_hours)
        self.weather.update(dt_hours)

        # -- snow accumulation / melt --
        if self.weather.state.current == Weather.SNOW:
            self.snow_depth_cm += self.weather.state.intensity * 0.5 * dt_hours
        elif self.weather.state.temperature > 2.0:
            self.snow_depth_cm = max(0.0, self.snow_depth_cm - 0.3 * dt_hours)

    # -- combined modifiers --

    def visibility(self) -> float:
        return WeatherEffects.combined_visibility(self.weather.state, self.time)

    def movement_speed_modifier(self) -> float:
        return WeatherEffects.movement_modifier(self.weather.state)

    # -- seasonal accessors --

    def foliage_state(self) -> float:
        """Green-cover fraction (1.0 when no seasonal cycle is attached)."""
        return self.seasonal.foliage_state() if self.seasonal is not None else 1.0

    def season(self) -> Season | None:
        """Current season, or None when no seasonal cycle is attached."""
        return self.seasonal.season() if self.seasonal is not None else None

    def seasonal_mobility_modifier(self) -> float:
        """Movement modifier further reduced by accumulated snow depth."""
        snow_penalty = min(0.4, self.snow_depth_cm * 0.02)
        return self.movement_speed_modifier() * (1.0 - snow_penalty)

    def accuracy_modifier(self) -> float:
        return WeatherEffects.accuracy_modifier(self.weather.state)

    def detection_range_modifier(self) -> float:
        """Combined visibility and sound modifier for detection range."""
        vis = self.visibility()
        snd = WeatherEffects.sound_modifier(self.weather.state)
        return (vis + snd) / 2.0

    # -- serialisation --

    def snapshot(self) -> dict:
        """JSON-serializable state dictionary.

        The base keys are byte-identical to the historical snapshot.  Seasonal
        keys are added *only* when a :class:`SeasonalCycle` is attached, so a
        default ``Environment()`` snapshot is unchanged (goldens depend on it).
        """
        ws = self.weather.state
        snap = {
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
        if self.seasonal is not None:
            snap["season"] = self.season().value
            snap["foliage_state"] = round(self.foliage_state(), 3)
            snap["snow_depth_cm"] = round(self.snow_depth_cm, 2)
            snap["seasonal_mobility_modifier"] = round(self.seasonal_mobility_modifier(), 3)
            snap["daylight_hours"] = round(self.seasonal.daylight_hours(), 2)
            snap["sunrise"] = round(self.seasonal.sunrise_hour(), 2)
            snap["sunset"] = round(self.seasonal.sunset_hour(), 2)
            snap["is_snow_season"] = self.seasonal.precip_is_snow()
        return snap

    def describe(self) -> str:
        """Human-readable one-liner, e.g. 'Clear day, light wind from NW, 22C'.

        Prefixed with the season (e.g. 'Winter: ...') when a seasonal cycle is
        attached.
        """
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
        base = f"{weather_label} {period}, {wind_desc}, {temp}"
        if self.seasonal is not None:
            return f"{self.seasonal.season().value.capitalize()}: {base}"
        return base

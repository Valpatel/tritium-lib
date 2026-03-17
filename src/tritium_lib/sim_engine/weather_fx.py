# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Weather visual effects system for Three.js-compatible rendering.

Builds on environment.py to generate detailed particle, shader, and
volumetric data that the frontend can render with Three.js or Canvas.

Systems:
  - RainSystem: raindrop particles with wind deflection and splash effects
  - SnowSystem: slow-falling snowflakes with accumulation tracking
  - FogSystem: 3D density grid for volumetric fog (raymarching / billboards)
  - LightningSystem: procedural bolt generation with branching + thunder
  - WindSystem: turbulent vector field affecting all particle systems
  - DayNightCycle: sky gradients, sun/moon position, star fields
  - WeatherFXEngine: unified tick producing a complete visual frame
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Bounds helper
# ---------------------------------------------------------------------------

@dataclass
class Bounds:
    """Axis-aligned bounding box for particle systems."""
    x_min: float = -50.0
    x_max: float = 50.0
    y_min: float = 0.0
    y_max: float = 50.0
    z_min: float = -50.0
    z_max: float = 50.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def depth(self) -> float:
        return self.z_max - self.z_min


# ---------------------------------------------------------------------------
# RainSystem
# ---------------------------------------------------------------------------

class RainSystem:
    """Generates raindrop particle data for Three.js instanced rendering.

    Drops fall with gravity and wind deflection. Splash effects are generated
    at ground level (y_min).
    """

    # Terminal velocity of rain varies by drop size; ~9 m/s for typical drops
    TERMINAL_VELOCITY: float = 9.0
    # Splash radius as fraction of drop speed
    SPLASH_RADIUS: float = 0.15

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate_rain(
        self,
        intensity: float,
        wind_dir: float,
        wind_speed: float,
        bounds: Bounds | None = None,
    ) -> dict:
        """Generate a frame of rain particle data.

        Parameters
        ----------
        intensity : float
            Rain intensity 0.0 (drizzle) to 1.0 (downpour).
        wind_dir : float
            Wind direction in radians (0 = +X).
        wind_speed : float
            Wind speed in m/s.
        bounds : Bounds, optional
            World-space region to fill with rain.

        Returns
        -------
        dict with keys:
            drop_count: int
            positions: list of [x, y, z]
            velocities: list of [vx, vy, vz]
            sizes: list of float (drop diameter)
            splashes: list of {position, radius, age}
        """
        b = bounds or Bounds()
        intensity = max(0.0, min(1.0, intensity))

        # Drop count scales with intensity and volume
        volume = b.width * b.height * b.depth
        density = 0.001 + 0.02 * intensity  # drops per cubic meter
        drop_count = max(1, int(volume * density))
        drop_count = min(drop_count, 10000)  # hard cap

        wind_vx = math.cos(wind_dir) * wind_speed
        wind_vz = math.sin(wind_dir) * wind_speed

        positions: list[list[float]] = []
        velocities: list[list[float]] = []
        sizes: list[float] = []
        splashes: list[dict] = []

        for _ in range(drop_count):
            x = self._rng.uniform(b.x_min, b.x_max)
            y = self._rng.uniform(b.y_min, b.y_max)
            z = self._rng.uniform(b.z_min, b.z_max)
            positions.append([x, y, z])

            # Fall speed with variation
            fall_speed = self.TERMINAL_VELOCITY * (0.7 + 0.6 * intensity)
            fall_speed *= self._rng.uniform(0.8, 1.2)

            vx = wind_vx * self._rng.uniform(0.6, 1.0)
            vy = -fall_speed
            vz = wind_vz * self._rng.uniform(0.6, 1.0)
            velocities.append([vx, vy, vz])

            # Drop size: 1-5mm, bigger in heavier rain
            size = self._rng.uniform(1.0, 2.0 + 3.0 * intensity)
            sizes.append(size)

            # Splashes at ground level
            if y < b.y_min + b.height * 0.1:
                splash_radius = size * self.SPLASH_RADIUS * fall_speed * 0.1
                splashes.append({
                    "position": [x, b.y_min, z],
                    "radius": splash_radius,
                    "age": self._rng.uniform(0.0, 0.3),
                })

        return {
            "drop_count": drop_count,
            "positions": positions,
            "velocities": velocities,
            "sizes": sizes,
            "splashes": splashes,
        }


# ---------------------------------------------------------------------------
# SnowSystem
# ---------------------------------------------------------------------------

class SnowSystem:
    """Generates snowflake particle data with slow drift and accumulation."""

    FALL_SPEED: float = 1.5  # m/s base

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._accumulation: float = 0.0  # cm of snow on ground

    @property
    def accumulation(self) -> float:
        """Current snow accumulation in cm."""
        return self._accumulation

    def generate_snow(
        self,
        intensity: float,
        wind_dir: float = 0.0,
        wind_speed: float = 0.0,
        bounds: Bounds | None = None,
    ) -> dict:
        """Generate a frame of snowflake particle data.

        Parameters
        ----------
        intensity : float
            Snow intensity 0.0 (flurries) to 1.0 (blizzard).
        wind_dir : float
            Wind direction in radians.
        wind_speed : float
            Wind speed in m/s.
        bounds : Bounds, optional
            World-space region.

        Returns
        -------
        dict with keys:
            flake_count, positions, velocities, sizes, rotations,
            accumulation_cm
        """
        b = bounds or Bounds()
        intensity = max(0.0, min(1.0, intensity))

        volume = b.width * b.height * b.depth
        density = 0.0005 + 0.008 * intensity
        flake_count = max(1, int(volume * density))
        flake_count = min(flake_count, 8000)

        wind_vx = math.cos(wind_dir) * wind_speed * 0.5
        wind_vz = math.sin(wind_dir) * wind_speed * 0.5

        positions: list[list[float]] = []
        velocities: list[list[float]] = []
        sizes: list[float] = []
        rotations: list[float] = []

        for _ in range(flake_count):
            x = self._rng.uniform(b.x_min, b.x_max)
            y = self._rng.uniform(b.y_min, b.y_max)
            z = self._rng.uniform(b.z_min, b.z_max)
            positions.append([x, y, z])

            # Slow drift with lateral oscillation
            fall = self.FALL_SPEED * (0.5 + intensity) * self._rng.uniform(0.6, 1.4)
            drift_x = wind_vx + self._rng.uniform(-0.3, 0.3)
            drift_z = wind_vz + self._rng.uniform(-0.3, 0.3)
            velocities.append([drift_x, -fall, drift_z])

            sizes.append(self._rng.uniform(2.0, 6.0 + 4.0 * intensity))
            rotations.append(self._rng.uniform(0.0, math.pi * 2))

        # Accumulation: ~1cm per hour at full intensity
        self._accumulation += intensity * 0.0003

        return {
            "flake_count": flake_count,
            "positions": positions,
            "velocities": velocities,
            "sizes": sizes,
            "rotations": rotations,
            "accumulation_cm": round(self._accumulation, 3),
        }


# ---------------------------------------------------------------------------
# FogSystem
# ---------------------------------------------------------------------------

class FogSystem:
    """Generates a 3D density grid for volumetric fog rendering.

    The grid can be used for raymarching in a shader or for placing
    billboard fog sprites at cell centers.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._phase: float = 0.0  # animation phase

    def generate_fog(
        self,
        density: float,
        wind_dir: float = 0.0,
        wind_speed: float = 0.0,
        bounds: Bounds | None = None,
        cell_size: float = 10.0,
    ) -> dict:
        """Generate a 3D fog density grid.

        Parameters
        ----------
        density : float
            Overall fog density 0.0 (clear) to 1.0 (pea soup).
        wind_dir : float
            Wind direction in radians, shifts fog over time.
        wind_speed : float
            Wind speed in m/s.
        bounds : Bounds, optional
            World region to cover.
        cell_size : float
            Grid cell edge length in meters.

        Returns
        -------
        dict with keys:
            grid_size: [nx, ny, nz]
            cell_size: float
            origin: [x, y, z]
            densities: flat list of floats (row-major, x varies fastest)
            wind_offset: [ox, oz] current wind displacement
        """
        b = bounds or Bounds()
        density = max(0.0, min(1.0, density))

        nx = max(1, int(b.width / cell_size))
        ny = max(1, int(b.height / cell_size))
        nz = max(1, int(b.depth / cell_size))

        # Wind shifts the noise pattern
        self._phase += wind_speed * 0.01
        wind_ox = math.cos(wind_dir) * self._phase
        wind_oz = math.sin(wind_dir) * self._phase

        densities: list[float] = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    # Simple pseudo-noise using sin products
                    wx = (ix + wind_ox) * 0.3
                    wy = iy * 0.5
                    wz = (iz + wind_oz) * 0.3

                    noise = (
                        math.sin(wx * 1.1 + wz * 0.7) * 0.4
                        + math.sin(wy * 0.9 + wx * 0.5) * 0.3
                        + math.sin(wz * 1.3 + wy * 0.6) * 0.3
                    )
                    # Normalize to 0-1 range and modulate by density
                    cell_density = max(0.0, (noise + 1.0) * 0.5 * density)

                    # Fog is denser near ground
                    height_ratio = iy / max(1, ny - 1)
                    ground_factor = 1.0 - height_ratio * 0.6
                    cell_density *= ground_factor

                    densities.append(round(cell_density, 4))

        return {
            "grid_size": [nx, ny, nz],
            "cell_size": cell_size,
            "origin": [b.x_min, b.y_min, b.z_min],
            "densities": densities,
            "wind_offset": [round(wind_ox, 3), round(wind_oz, 3)],
        }


# ---------------------------------------------------------------------------
# LightningSystem
# ---------------------------------------------------------------------------

class LightningSystem:
    """Procedural lightning bolt generation with branching.

    Bolts are lists of line segments that can be rendered as GL_LINES
    or as tube geometry in Three.js.
    """

    SPEED_OF_SOUND: float = 343.0  # m/s

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate_bolt(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        branches: int = 3,
        jitter: float = 0.3,
    ) -> dict:
        """Generate a branching lightning bolt between two points.

        Parameters
        ----------
        start : (x, y, z)
            Bolt origin (usually high in sky).
        end : (x, y, z)
            Bolt terminus (usually ground level).
        branches : int
            Number of side branches to generate.
        jitter : float
            Lateral displacement factor (fraction of segment length).

        Returns
        -------
        dict with keys:
            segments: list of [[x1,y1,z1], [x2,y2,z2]] line segments
            branch_count: int
            brightness: float 0-1
        """
        segments = self._subdivide_bolt(start, end, jitter, depth=5)

        # Add branches from random midpoints
        branch_segments: list[list[list[float]]] = []
        if branches > 0 and len(segments) >= 2:
            branch_points = self._rng.sample(
                range(len(segments)),
                min(branches, len(segments)),
            )
            for bp_idx in branch_points:
                seg = segments[bp_idx]
                branch_start = seg[0]
                # Branch end diverges laterally
                dx = (seg[1][0] - seg[0][0])
                dy = (seg[1][1] - seg[0][1])
                dz = (seg[1][2] - seg[0][2])
                length = math.sqrt(dx * dx + dy * dy + dz * dz)
                branch_len = length * self._rng.uniform(1.0, 3.0)

                branch_end = [
                    branch_start[0] + dx * 0.5 + self._rng.uniform(-1, 1) * branch_len,
                    branch_start[1] + dy * 0.5 + abs(self._rng.gauss(0, branch_len * 0.3)),
                    branch_start[2] + dz * 0.5 + self._rng.uniform(-1, 1) * branch_len,
                ]
                sub = self._subdivide_bolt(
                    tuple(branch_start), tuple(branch_end),
                    jitter * 1.5, depth=3,
                )
                branch_segments.extend(sub)

        all_segments = segments + branch_segments

        return {
            "segments": all_segments,
            "branch_count": len(branch_segments),
            "brightness": self._rng.uniform(0.8, 1.0),
        }

    def _subdivide_bolt(
        self,
        start: tuple,
        end: tuple,
        jitter: float,
        depth: int,
    ) -> list[list[list[float]]]:
        """Recursively subdivide a line segment with random displacement."""
        if depth <= 0:
            return [[[*start], [*end]]]

        mx = (start[0] + end[0]) * 0.5 + self._rng.gauss(0, jitter * abs(end[0] - start[0] + 0.1))
        my = (start[1] + end[1]) * 0.5 + self._rng.gauss(0, jitter * abs(end[1] - start[1] + 0.1))
        mz = (start[2] + end[2]) * 0.5 + self._rng.gauss(0, jitter * abs(end[2] - start[2] + 0.1))
        mid = (mx, my, mz)

        left = self._subdivide_bolt(start, mid, jitter * 0.7, depth - 1)
        right = self._subdivide_bolt(mid, end, jitter * 0.7, depth - 1)
        return left + right

    def strike(
        self,
        position: tuple[float, float, float],
        observer: tuple[float, float, float] = (0.0, 0.0, 0.0),
        cloud_height: float = 2000.0,
    ) -> dict:
        """Generate a lightning strike with flash and thunder delay.

        Parameters
        ----------
        position : (x, y, z)
            Ground strike position.
        observer : (x, y, z)
            Observer position for thunder delay calculation.
        cloud_height : float
            Height of cloud base in meters.

        Returns
        -------
        dict with keys:
            bolt: dict from generate_bolt
            flash_intensity: float 0-1
            thunder_delay_s: float (seconds until thunder reaches observer)
            rumble_duration_s: float
        """
        start = (position[0], cloud_height, position[2])
        bolt = self.generate_bolt(start, position, branches=self._rng.randint(2, 5))

        dx = position[0] - observer[0]
        dy = position[1] - observer[1]
        dz = position[2] - observer[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        thunder_delay = distance / self.SPEED_OF_SOUND
        rumble_duration = 0.5 + distance * 0.002 + self._rng.uniform(0.0, 1.0)

        return {
            "bolt": bolt,
            "flash_intensity": self._rng.uniform(0.7, 1.0),
            "thunder_delay_s": round(thunder_delay, 3),
            "rumble_duration_s": round(rumble_duration, 2),
        }


# ---------------------------------------------------------------------------
# WindSystem
# ---------------------------------------------------------------------------

class WindSystem:
    """Turbulent wind field with gusts for particle and object influence.

    Provides smooth wind vectors at any position/time using layered
    sinusoidal turbulence over a base direction.
    """

    def __init__(
        self,
        base_dir: float = 0.0,
        base_speed: float = 5.0,
        gust_strength: float = 0.3,
        turbulence: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.base_speed = base_speed
        self.gust_strength = gust_strength
        self.turbulence = turbulence
        self._rng = random.Random(seed)
        self._time: float = 0.0
        # Random phase offsets for turbulence layers
        self._phases = [self._rng.uniform(0, 100) for _ in range(6)]

    def advance(self, dt: float) -> None:
        """Advance internal clock for gust evolution."""
        self._time += dt

    def get_wind_at(
        self,
        position: tuple[float, float] | tuple[float, float, float],
        time: float | None = None,
    ) -> tuple[float, float]:
        """Get wind vector (vx, vz) at a world position and time.

        Parameters
        ----------
        position : (x, z) or (x, y, z)
            World-space query point.
        time : float, optional
            Override internal time.

        Returns
        -------
        (vx, vz) wind velocity in m/s.
        """
        t = time if time is not None else self._time
        px = position[0]
        pz = position[-1]  # last component is z whether 2D or 3D

        # Base wind
        bx = math.cos(self.base_dir) * self.base_speed
        bz = math.sin(self.base_dir) * self.base_speed

        # Gust: slow large-scale oscillation
        gust_factor = 1.0 + self.gust_strength * math.sin(
            t * 0.3 + self._phases[0]
        ) * math.sin(t * 0.17 + self._phases[1])

        # Turbulence: position-dependent small variation
        turb_x = self.turbulence * self.base_speed * (
            math.sin(px * 0.05 + t * 0.7 + self._phases[2]) * 0.6
            + math.sin(pz * 0.08 + t * 0.5 + self._phases[3]) * 0.4
        )
        turb_z = self.turbulence * self.base_speed * (
            math.sin(pz * 0.06 + t * 0.6 + self._phases[4]) * 0.5
            + math.sin(px * 0.07 + t * 0.4 + self._phases[5]) * 0.5
        )

        vx = bx * gust_factor + turb_x
        vz = bz * gust_factor + turb_z

        return (vx, vz)

    def to_three_js(
        self,
        bounds: Bounds | None = None,
        resolution: int = 8,
    ) -> dict:
        """Export a wind vector field grid for Three.js particle systems.

        Parameters
        ----------
        bounds : Bounds, optional
            Area to sample.
        resolution : int
            Grid points per axis.

        Returns
        -------
        dict with keys:
            resolution: int
            origin: [x, z]
            cell_size: [dx, dz]
            vectors: list of [vx, vz] in row-major order (z varies fastest)
            time: float
        """
        b = bounds or Bounds()
        dx = b.width / max(1, resolution - 1)
        dz = b.depth / max(1, resolution - 1)

        vectors: list[list[float]] = []
        for iz in range(resolution):
            for ix in range(resolution):
                px = b.x_min + ix * dx
                pz = b.z_min + iz * dz
                vx, vz = self.get_wind_at((px, pz))
                vectors.append([round(vx, 3), round(vz, 3)])

        return {
            "resolution": resolution,
            "origin": [b.x_min, b.z_min],
            "cell_size": [round(dx, 3), round(dz, 3)],
            "vectors": vectors,
            "time": round(self._time, 3),
        }


# ---------------------------------------------------------------------------
# DayNightCycle
# ---------------------------------------------------------------------------

# Sky color palettes by time segment
_SKY_COLORS: dict[str, dict[str, list[str]]] = {
    "night": {
        "top": ["#020210", "#050520", "#080830"],
        "middle": ["#0a0a1a", "#0e0e2a", "#10103a"],
        "bottom": ["#0c0c1e", "#121232", "#181848"],
    },
    "dawn": {
        "top": ["#1a1040", "#2e1860", "#4a2080"],
        "middle": ["#803050", "#c04860", "#e06848"],
        "bottom": ["#e08040", "#f0a040", "#ffc040"],
    },
    "day": {
        "top": ["#1060d0", "#2080e0", "#3090e8"],
        "middle": ["#40a0f0", "#60b8f8", "#80c8ff"],
        "bottom": ["#90d0ff", "#b0e0ff", "#d0f0ff"],
    },
    "dusk": {
        "top": ["#301060", "#4a1870", "#602080"],
        "middle": ["#a03050", "#c04848", "#d06030"],
        "bottom": ["#e08020", "#f0a010", "#ffc000"],
    },
}

# Star catalog: bright stars with RA-like positions
_STAR_CATALOG: list[dict] = [
    {"name": "Sirius", "ra": 1.77, "dec": -0.29, "mag": -1.46},
    {"name": "Canopus", "ra": 1.68, "dec": -0.92, "mag": -0.74},
    {"name": "Arcturus", "ra": 3.73, "dec": 0.33, "mag": -0.05},
    {"name": "Vega", "ra": 4.87, "dec": 0.68, "mag": 0.03},
    {"name": "Capella", "ra": 1.38, "dec": 0.80, "mag": 0.08},
    {"name": "Rigel", "ra": 1.37, "dec": -0.14, "mag": 0.13},
    {"name": "Procyon", "ra": 2.00, "dec": 0.09, "mag": 0.34},
    {"name": "Betelgeuse", "ra": 1.55, "dec": 0.13, "mag": 0.42},
    {"name": "Altair", "ra": 5.13, "dec": 0.15, "mag": 0.76},
    {"name": "Aldebaran", "ra": 1.20, "dec": 0.29, "mag": 0.85},
    {"name": "Spica", "ra": 3.51, "dec": -0.19, "mag": 1.04},
    {"name": "Antares", "ra": 4.32, "dec": -0.46, "mag": 1.06},
    {"name": "Pollux", "ra": 2.03, "dec": 0.49, "mag": 1.14},
    {"name": "Fomalhaut", "ra": 5.96, "dec": -0.52, "mag": 1.16},
    {"name": "Deneb", "ra": 5.42, "dec": 0.78, "mag": 1.25},
    {"name": "Regulus", "ra": 2.65, "dec": 0.21, "mag": 1.35},
]


class DayNightCycle:
    """Detailed sky state for Three.js scene lighting and backgrounds.

    Tracks sun/moon positions, sky color gradients, and visible stars.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def get_sky_gradient(self, hour: float) -> list[str]:
        """Get sky color gradient [top, middle, bottom] as hex strings.

        Parameters
        ----------
        hour : float
            Time in hours 0-24.

        Returns
        -------
        list of 3 hex color strings: [top, middle, bottom].
        """
        hour = hour % 24.0

        if 6.5 <= hour <= 17.5:
            segment = "day"
            t = (hour - 6.5) / 11.0
        elif 5.0 <= hour < 6.5:
            segment = "dawn"
            t = (hour - 5.0) / 1.5
        elif 17.5 < hour <= 19.5:
            segment = "dusk"
            t = (hour - 17.5) / 2.0
        else:
            segment = "night"
            if hour > 19.5:
                t = min(1.0, (hour - 19.5) / 3.0)
            else:
                t = max(0.0, 1.0 - hour / 5.0)

        colors = _SKY_COLORS[segment]
        t = max(0.0, min(1.0, t))

        def pick(arr: list[str]) -> str:
            if len(arr) == 1:
                return arr[0]
            pos = t * (len(arr) - 1)
            idx = min(int(pos), len(arr) - 2)
            return arr[idx]

        return [pick(colors["top"]), pick(colors["middle"]), pick(colors["bottom"])]

    def get_sun_position(self, hour: float) -> tuple[float, float, float]:
        """Get sun position as (x, y, z) for a directional light.

        The sun traces an arc: rises in +X, peaks at +Y, sets in -X.
        Y is up. Returned vector is unit-length when sun is above horizon.

        Parameters
        ----------
        hour : float
            Time in hours 0-24.

        Returns
        -------
        (x, y, z) sun direction vector.
        """
        hour = hour % 24.0
        # Sun arc: 6am = horizon east, 12pm = zenith, 6pm = horizon west
        angle = (hour - 6.0) * math.pi / 12.0  # 0 at 6am, pi at 6pm

        y = math.sin(angle)
        x = math.cos(angle)
        # Slight z offset for visual depth
        z = 0.2 * math.sin(angle * 0.5)

        # Normalize
        length = math.sqrt(x * x + y * y + z * z)
        if length > 0.001:
            x /= length
            y /= length
            z /= length

        return (round(x, 4), round(y, 4), round(z, 4))

    def get_moon_phase(self, day: float) -> float:
        """Get moon phase as 0-1 (0 = new moon, 0.5 = full moon, 1 = new again).

        Parameters
        ----------
        day : float
            Day number (fractional).

        Returns
        -------
        float 0-1 representing the lunar cycle.
        """
        # Synodic period ~29.53 days
        return (day % 29.53) / 29.53

    def get_moon_position(self, hour: float, day: float = 0.0) -> tuple[float, float, float]:
        """Get moon position. Roughly opposite the sun, shifted by phase.

        Returns
        -------
        (x, y, z) moon direction vector.
        """
        hour = hour % 24.0
        phase_offset = self.get_moon_phase(day) * math.pi * 2
        angle = (hour - 18.0) * math.pi / 12.0 + phase_offset * 0.1

        y = math.sin(angle)
        x = -math.cos(angle)
        z = 0.15 * math.cos(angle * 0.7)

        length = math.sqrt(x * x + y * y + z * z)
        if length > 0.001:
            x /= length
            y /= length
            z /= length

        return (round(x, 4), round(y, 4), round(z, 4))

    def get_star_field(self, hour: float, min_magnitude: float = 2.0) -> list[dict]:
        """Get visible stars for the current time of day.

        Stars are visible when light level is low enough. Returns
        positions on a unit sphere suitable for a skybox or point cloud.

        Parameters
        ----------
        hour : float
            Time in hours 0-24.
        min_magnitude : float
            Faintest star magnitude to include (lower = fewer stars).

        Returns
        -------
        list of dicts with: name, x, y, z, magnitude, brightness
        """
        hour = hour % 24.0

        # Stars fade based on sky brightness
        # Full visibility at night (21-5), invisible during day (7-17)
        if 7.0 <= hour <= 17.0:
            return []

        if hour > 17.0:
            sky_brightness = max(0.0, (hour - 17.0) / 4.0)
        elif hour < 7.0:
            sky_brightness = max(0.0, (7.0 - hour) / 2.0)
        else:
            sky_brightness = 0.0

        sky_brightness = min(1.0, sky_brightness)

        # Hour angle rotation (sky rotates ~15 deg/hour)
        rotation = hour * math.pi / 12.0

        stars: list[dict] = []
        for s in _STAR_CATALOG:
            if s["mag"] > min_magnitude:
                continue

            # Position on unit sphere using RA and Dec
            ra = s["ra"] + rotation
            dec = s["dec"]
            x = math.cos(dec) * math.cos(ra)
            y = math.sin(dec)
            z = math.cos(dec) * math.sin(ra)

            # Skip below-horizon stars
            if y < -0.05:
                continue

            # Brightness: brighter stars (lower magnitude) are more visible
            brightness = sky_brightness * max(0.0, 1.0 - s["mag"] / min_magnitude)

            if brightness > 0.05:
                stars.append({
                    "name": s["name"],
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "z": round(z, 4),
                    "magnitude": s["mag"],
                    "brightness": round(brightness, 3),
                })

        return stars


# ---------------------------------------------------------------------------
# WeatherFXEngine
# ---------------------------------------------------------------------------

class WeatherFXEngine:
    """Unified weather visual effects engine.

    Combines all weather subsystems and produces a complete visual
    frame dict suitable for sending to the Three.js frontend via
    WebSocket or API.
    """

    def __init__(self, seed: int | None = None) -> None:
        base_seed = seed if seed is not None else random.randint(0, 2**31)
        self.rain = RainSystem(seed=base_seed)
        self.snow = SnowSystem(seed=base_seed + 1)
        self.fog = FogSystem(seed=base_seed + 2)
        self.lightning = LightningSystem(seed=base_seed + 3)
        self.wind = WindSystem(seed=base_seed + 4)
        self.day_night = DayNightCycle(seed=base_seed + 5)
        self._time: float = 0.0

    def tick(
        self,
        dt: float,
        weather_state: dict,
        time_state: dict,
        bounds: Bounds | None = None,
    ) -> dict:
        """Produce a complete visual weather frame.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        weather_state : dict
            Must have: weather (str), intensity (float),
            wind_speed (float), wind_direction (float in degrees).
        time_state : dict
            Must have: hour (float). Optional: day (float).
        bounds : Bounds, optional
            World region for particle generation.

        Returns
        -------
        dict with all visual weather data for the frame.
        """
        self._time += dt
        b = bounds or Bounds()

        weather = weather_state.get("weather", "clear")
        intensity = float(weather_state.get("intensity", 0.0))
        wind_speed = float(weather_state.get("wind_speed", 0.0))
        wind_dir_deg = float(weather_state.get("wind_direction", 0.0))
        wind_dir = math.radians(wind_dir_deg)

        hour = float(time_state.get("hour", 12.0))
        day = float(time_state.get("day", 0.0))

        # Update wind system
        self.wind.base_dir = wind_dir
        self.wind.base_speed = wind_speed
        self.wind.advance(dt)

        frame: dict = {
            "time": round(self._time, 3),
            "hour": round(hour, 2),
            "sky": {
                "gradient": self.day_night.get_sky_gradient(hour),
                "sun": self.day_night.get_sun_position(hour),
                "moon": self.day_night.get_moon_position(hour, day),
                "moon_phase": round(self.day_night.get_moon_phase(day), 3),
                "stars": self.day_night.get_star_field(hour),
            },
            "wind": self.wind.to_three_js(bounds=b),
        }

        # Rain
        if weather in ("rain", "heavy_rain", "storm"):
            rain_intensity = intensity
            if weather == "heavy_rain":
                rain_intensity = max(intensity, 0.7)
            elif weather == "storm":
                rain_intensity = max(intensity, 0.8)
            frame["rain"] = self.rain.generate_rain(
                rain_intensity, wind_dir, wind_speed, b,
            )
        else:
            frame["rain"] = None

        # Snow
        if weather == "snow":
            frame["snow"] = self.snow.generate_snow(
                intensity, wind_dir, wind_speed, b,
            )
        else:
            frame["snow"] = None

        # Fog
        if weather in ("fog", "storm", "heavy_rain"):
            fog_density = intensity * 0.8
            if weather == "fog":
                fog_density = max(intensity, 0.5)
            frame["fog"] = self.fog.generate_fog(
                fog_density, wind_dir, wind_speed, b,
            )
        else:
            frame["fog"] = None

        # Lightning (only during storms, random chance per tick)
        if weather == "storm" and self._rng_check(intensity * 0.1 * dt):
            strike_x = self._rng_pos(b.x_min, b.x_max)
            strike_z = self._rng_pos(b.z_min, b.z_max)
            frame["lightning"] = self.lightning.strike(
                (strike_x, b.y_min, strike_z),
            )
        else:
            frame["lightning"] = None

        return frame

    def _rng_check(self, probability: float) -> bool:
        """Random check using the lightning system's RNG."""
        return self.lightning._rng.random() < probability

    def _rng_pos(self, lo: float, hi: float) -> float:
        """Random position using the lightning system's RNG."""
        return self.lightning._rng.uniform(lo, hi)

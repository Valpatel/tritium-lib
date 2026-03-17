# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Three.js render bridge for the sim engine.

Converts sim_engine state (units, projectiles, effects, terrain, weather)
into JSON-serializable frame dicts that the Three.js frontend can render
directly.  The SimRenderer composes individual layer renderers and supports
delta-only frames for bandwidth efficiency.

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2

# ---------------------------------------------------------------------------
# Render layers
# ---------------------------------------------------------------------------


class RenderLayer(Enum):
    """Selectable rendering layers."""

    UNITS = "units"
    PROJECTILES = "projectiles"
    EFFECTS = "effects"
    TERRAIN = "terrain"
    CROWD = "crowd"
    WEATHER = "weather"
    UI = "ui"
    DEBUG = "debug"


# ---------------------------------------------------------------------------
# Color utilities
# ---------------------------------------------------------------------------

_ALLIANCE_COLORS: dict[str, str] = {
    "friendly": "#05ffa1",
    "hostile": "#ff2a6d",
    "neutral": "#00f0ff",
    "unknown": "#fcee0a",
}

_MOOD_COLORS: dict[str, str] = {
    "calm": "#05ffa1",
    "agitated": "#fcee0a",
    "rioting": "#ff2a6d",
    "panicked": "#ff6600",
    "fleeing": "#888888",
}

_TRACER_COLORS: dict[str, str] = {
    "bullet": "#ffaa00",
    "laser": "#ff0000",
    "plasma": "#00ccff",
    "missile": "#ff6600",
    "grenade": "#88ff00",
    "shotgun": "#ffdd44",
    "sniper": "#ffffff",
    "smg": "#ffcc00",
}


def alliance_color(alliance: str) -> str:
    """Return the hex color for an alliance string."""
    return _ALLIANCE_COLORS.get(alliance.lower(), _ALLIANCE_COLORS["unknown"])


def mood_color(mood: str) -> str:
    """Return the hex color for a civilian mood."""
    return _MOOD_COLORS.get(mood.lower(), _MOOD_COLORS["calm"])


def damage_flash(intensity: float) -> str:
    """Interpolate white (#ffffff) to red (#ff0000) based on damage intensity.

    *intensity* is clamped to [0, 1].  0.0 = white, 1.0 = pure red.
    """
    intensity = max(0.0, min(1.0, intensity))
    # Interpolate green and blue channels from 0xff down to 0x00.
    g = int(255 * (1.0 - intensity))
    b = g
    return f"#{255:02x}{g:02x}{b:02x}"


def tracer_color(weapon_type: str) -> str:
    """Return a tracer/projectile color for a weapon type."""
    return _TRACER_COLORS.get(weapon_type.lower(), _TRACER_COLORS["bullet"])


# ---------------------------------------------------------------------------
# NATO marker helpers
# ---------------------------------------------------------------------------

_MARKER_MAP: dict[str, str] = {
    "infantry": "nato_infantry",
    "sniper": "nato_infantry",
    "heavy": "nato_heavy",
    "medic": "nato_medic",
    "engineer": "nato_engineer",
    "scout": "nato_recon",
    "vehicle": "nato_vehicle",
    "drone": "nato_drone",
    "turret": "nato_turret",
    "civilian": "circle",
}


def _marker_type(unit_type: str) -> str:
    return _MARKER_MAP.get(unit_type.lower(), "circle")


# ---------------------------------------------------------------------------
# Unit renderer
# ---------------------------------------------------------------------------


class UnitRenderer:
    """Convert unit dicts into Three.js-ready render data."""

    @staticmethod
    def render_units(
        units: list[dict[str, Any]],
        camera_pos: Vec2 | None = None,
    ) -> list[dict[str, Any]]:
        """Return a list of Three.js-ready unit render dicts.

        Each input unit dict is expected to have at minimum:
            id, x, y, type, alliance

        Optional fields: z, heading, health, max_health, status, label,
        effects, scale.
        """
        result: list[dict[str, Any]] = []
        for u in units:
            uid = u.get("id", "")
            utype = str(u.get("type", "infantry")).lower()
            ualliance = str(u.get("alliance", "unknown")).lower()
            max_hp = float(u.get("max_health", 100.0))
            cur_hp = float(u.get("health", max_hp))
            health_ratio = cur_hp / max_hp if max_hp > 0 else 0.0

            entry: dict[str, Any] = {
                "id": uid,
                "x": float(u.get("x", 0.0)),
                "y": float(u.get("y", 0.0)),
                "z": float(u.get("z", 0.0)),
                "heading": float(u.get("heading", 0.0)),
                "type": utype,
                "alliance": ualliance,
                "color": alliance_color(ualliance),
                "health": round(health_ratio, 4),
                "status": str(u.get("status", "idle")),
                "scale": float(u.get("scale", 1.0)),
                "effects": list(u.get("effects", [])),
                "label": str(u.get("label", uid)),
                "marker_type": _marker_type(utype),
            }
            result.append(entry)
        return result


# ---------------------------------------------------------------------------
# Projectile renderer
# ---------------------------------------------------------------------------


class ProjectileRenderer:
    """Convert projectile dicts into Three.js-ready render data."""

    @staticmethod
    def render_projectiles(
        projectiles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for p in projectiles:
            ptype = str(p.get("type", "bullet")).lower()
            entry: dict[str, Any] = {
                "id": p.get("id", ""),
                "x": float(p.get("x", 0.0)),
                "y": float(p.get("y", 0.0)),
                "z": float(p.get("z", 0.5)),
                "vx": float(p.get("vx", 0.0)),
                "vy": float(p.get("vy", 0.0)),
                "vz": float(p.get("vz", 0.0)),
                "type": ptype,
                "color": tracer_color(ptype),
                "trail_length": float(p.get("trail_length", 3.0)),
                "glow": bool(p.get("glow", True)),
            }
            result.append(entry)
        return result


# ---------------------------------------------------------------------------
# Effect renderer
# ---------------------------------------------------------------------------

_EFFECT_DEFAULTS: dict[str, dict[str, Any]] = {
    "explosion": {
        "radius": 5.0,
        "intensity": 1.0,
        "color": "#ff4400",
        "particle_count": 50,
        "duration": 0.5,
    },
    "smoke": {
        "radius": 8.0,
        "opacity": 0.6,
        "color": "#888888",
        "billboards": True,
    },
    "fire": {
        "radius": 3.0,
        "intensity": 0.8,
        "color": "#ff6600",
        "emitter": {"rate": 100, "lifetime": 1.5, "speed": 2.0},
    },
    "flashbang": {
        "radius": 15.0,
        "intensity": 1.0,
        "color": "#ffffff",
        "duration": 0.3,
        "screen_flash": True,
    },
    "muzzle_flash": {
        "radius": 1.0,
        "intensity": 0.9,
        "color": "#ffcc00",
        "duration": 0.05,
    },
    "blood": {
        "radius": 0.5,
        "intensity": 0.7,
        "color": "#cc0000",
        "particle_count": 20,
        "duration": 0.3,
    },
}


class EffectRenderer:
    """Convert effect dicts into Three.js-ready render data."""

    @staticmethod
    def render_effects(
        effects: list[dict[str, Any]],
        particles: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for e in effects:
            etype = str(e.get("type", "explosion")).lower()
            defaults = _EFFECT_DEFAULTS.get(etype, _EFFECT_DEFAULTS["explosion"])

            entry: dict[str, Any] = {"type": etype}
            entry["x"] = float(e.get("x", 0.0))
            entry["y"] = float(e.get("y", 0.0))
            entry["z"] = float(e.get("z", 0.0))

            # Merge defaults, then override with explicit values
            for key, default_val in defaults.items():
                entry[key] = e.get(key, default_val)

            # Age tracking
            entry["age"] = float(e.get("age", 0.0))

            result.append(entry)

        # Append raw particles if provided
        if particles:
            for p in particles:
                result.append({
                    "type": "particle",
                    "x": float(p.get("x", 0.0)),
                    "y": float(p.get("y", 0.0)),
                    "z": float(p.get("z", 0.0)),
                    "color": str(p.get("color", "#ffffff")),
                    "size": float(p.get("size", 1.0)),
                    "opacity": float(p.get("opacity", 1.0)),
                    "age": float(p.get("age", 0.0)),
                })

        return result


# ---------------------------------------------------------------------------
# Weather renderer
# ---------------------------------------------------------------------------

# Sky color interpolation: night -> dawn/dusk -> day
_SKY_NIGHT = (0x0A, 0x0A, 0x1E)     # #0a0a1e
_SKY_TWILIGHT = (0x2E, 0x1A, 0x3E)  # #2e1a3e
_SKY_DAY = (0x44, 0x88, 0xCC)       # #4488cc
_SKY_OVERCAST = (0x55, 0x55, 0x66)  # #555566


def _lerp_color(
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
    t: float,
) -> str:
    t = max(0.0, min(1.0, t))
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _sun_angle(hour: float) -> float:
    """Sun elevation angle in degrees. Negative = below horizon."""
    # Simple sine model: peaks at noon (90 deg), 0 at 6am/6pm
    return 90.0 * math.sin((hour - 6.0) * math.pi / 12.0)


class WeatherRenderer:
    """Convert weather/time state into Three.js environment params."""

    @staticmethod
    def render_weather(
        weather_state: dict[str, Any],
        time_state: dict[str, Any],
    ) -> dict[str, Any]:
        hour = float(time_state.get("hour", 12.0))
        overcast = float(weather_state.get("overcast", 0.0))

        # Sky color based on time of day
        if hour >= 21.0 or hour <= 5.0:
            # Night
            sky = _SKY_NIGHT
            ambient = 0.1
        elif 5.0 < hour < 6.0:
            # Dawn
            t = hour - 5.0
            sky_base = _lerp_color(_SKY_NIGHT, _SKY_TWILIGHT, t)
            ambient = 0.1 + 0.2 * t
            sky = None  # use string directly
        elif 18.0 < hour < 21.0:
            # Dusk
            t = (hour - 18.0) / 3.0
            sky_base = _lerp_color(_SKY_DAY, _SKY_NIGHT, t)
            ambient = 0.8 - 0.7 * t
            sky = None
        else:
            # Day
            sky = _SKY_DAY
            ambient = 0.8

        if sky is not None:
            sky_color = f"#{sky[0]:02x}{sky[1]:02x}{sky[2]:02x}"
        else:
            sky_color = sky_base  # type: ignore[possibly-undefined]

        # Apply overcast darkening
        if overcast > 0:
            sky_color = _lerp_color(
                tuple(int(sky_color[i:i+2], 16) for i in (1, 3, 5)),  # type: ignore[arg-type]
                _SKY_OVERCAST,
                overcast,
            )
            ambient *= (1.0 - 0.3 * overcast)

        # Sun
        sun_ang = _sun_angle(hour)
        sun_intensity = max(0.0, sun_ang / 90.0) * (1.0 - 0.5 * overcast)

        result: dict[str, Any] = {
            "sky_color": sky_color,
            "ambient_light": round(ambient, 3),
            "fog_density": round(float(weather_state.get("fog", 0.0)), 4),
            "wind": {
                "speed": float(weather_state.get("wind_speed", 0.0)),
                "direction": float(weather_state.get("wind_direction", 0.0)),
            },
            "sun": {
                "angle": round(sun_ang, 2),
                "color": "#ffdd88",
                "intensity": round(sun_intensity, 3),
            },
        }

        # Rain
        rain_intensity = float(weather_state.get("rain", 0.0))
        if rain_intensity > 0:
            wind_dir = float(weather_state.get("wind_direction", 0.0))
            wind_spd = float(weather_state.get("wind_speed", 0.0))
            result["rain"] = {
                "intensity": rain_intensity,
                "direction": [
                    round(math.sin(wind_dir) * wind_spd * 0.02, 4),
                    -1.0,
                    round(math.cos(wind_dir) * wind_spd * 0.02, 4),
                ],
                "color": "#aaccff",
            }
            result["particles"] = {
                "type": "rain",
                "count": int(rain_intensity * 3000),
                "speed": 8.0 + wind_spd * 0.5,
            }
            # Rain increases fog slightly
            result["fog_density"] = round(
                result["fog_density"] + rain_intensity * 0.01, 4
            )

        # Snow
        snow_intensity = float(weather_state.get("snow", 0.0))
        if snow_intensity > 0:
            result["snow"] = {
                "intensity": snow_intensity,
                "color": "#ffffff",
            }
            result["particles"] = {
                "type": "snow",
                "count": int(snow_intensity * 2000),
                "speed": 2.0,
            }

        return result


# ---------------------------------------------------------------------------
# Terrain renderer
# ---------------------------------------------------------------------------


class TerrainRenderer:
    """Convert terrain data into Three.js mesh and overlay data."""

    @staticmethod
    def render_terrain(
        heightmap_data: list[list[float]],
        cell_size: float,
        cover_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a Three.js-compatible terrain mesh from a 2D heightmap.

        Returns vertices (flat list of x, y, z triples), face indices, vertex
        colors (terrain-height-based), and an optional cover overlay.
        """
        if not heightmap_data or not heightmap_data[0]:
            return {"vertices": [], "faces": [], "colors": [], "cover_overlay": []}

        rows = len(heightmap_data)
        cols = len(heightmap_data[0])

        vertices: list[float] = []
        colors: list[str] = []

        # Find height range for color mapping
        flat = [h for row in heightmap_data for h in row]
        h_min = min(flat) if flat else 0.0
        h_max = max(flat) if flat else 1.0
        h_range = h_max - h_min if h_max > h_min else 1.0

        for r in range(rows):
            for c in range(cols):
                x = c * cell_size
                y = r * cell_size
                z = heightmap_data[r][c]
                vertices.extend([x, y, z])

                # Color: low=green, high=brown/grey
                t = (z - h_min) / h_range
                gr = int(120 + 80 * (1.0 - t))
                rd = int(80 + 100 * t)
                bl = int(60 + 40 * (1.0 - t))
                colors.append(f"#{rd:02x}{gr:02x}{bl:02x}")

        # Triangle faces (two per grid cell)
        faces: list[int] = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                i = r * cols + c
                # Triangle 1
                faces.extend([i, i + cols, i + 1])
                # Triangle 2
                faces.extend([i + 1, i + cols, i + cols + 1])

        result: dict[str, Any] = {
            "vertices": vertices,
            "faces": faces,
            "colors": colors,
        }

        # Cover overlay
        if cover_data:
            overlay: list[dict[str, Any]] = []
            cells = cover_data.get("cells", {})
            for key, value in cells.items():
                # key expected as "x,y" string or tuple
                if isinstance(key, str):
                    parts = key.split(",")
                    cx, cy = float(parts[0]), float(parts[1])
                else:
                    cx, cy = float(key[0]), float(key[1])
                v = float(value)
                # Alpha scales with cover value
                alpha = int(v * 255)
                overlay.append({
                    "x": cx,
                    "y": cy,
                    "value": v,
                    "color": f"#00ff00{alpha:02x}",
                })
            result["cover_overlay"] = overlay
        else:
            result["cover_overlay"] = []

        return result

    @staticmethod
    def render_los_overlay(
        visible_cells: set[tuple[int, int]],
        grid_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Generate a fog-of-war overlay for line-of-sight.

        Returns a flat grid where each cell is either visible (1) or fogged (0).
        """
        width, height = grid_size
        fog: list[list[int]] = []
        for r in range(height):
            row: list[int] = []
            for c in range(width):
                row.append(1 if (c, r) in visible_cells else 0)
            fog.append(row)

        return {
            "width": width,
            "height": height,
            "fog": fog,
            "fog_color": "#000000",
            "fog_opacity": 0.7,
        }


# ---------------------------------------------------------------------------
# Crowd renderer (civilians / NPCs)
# ---------------------------------------------------------------------------


class CrowdRenderer:
    """Render crowd/civilian entities."""

    @staticmethod
    def render_crowd(crowd: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for c in crowd:
            cmood = str(c.get("mood", "calm")).lower()
            entry: dict[str, Any] = {
                "id": c.get("id", ""),
                "x": float(c.get("x", 0.0)),
                "y": float(c.get("y", 0.0)),
                "z": float(c.get("z", 0.0)),
                "mood": cmood,
                "color": mood_color(cmood),
                "heading": float(c.get("heading", 0.0)),
                "speed": float(c.get("speed", 0.0)),
                "scale": float(c.get("scale", 0.8)),
            }
            result.append(entry)
        return result


# ---------------------------------------------------------------------------
# SimRenderer — combined frame renderer
# ---------------------------------------------------------------------------

_ALL_LAYERS = set(RenderLayer)


class SimRenderer:
    """Main renderer that combines all layer renderers into a single frame.

    Parameters
    ----------
    layers : set of RenderLayer, optional
        Which layers to include.  ``None`` means all layers.
    """

    def __init__(self, layers: set[RenderLayer] | None = None) -> None:
        self.layers: set[RenderLayer] = layers if layers is not None else set(_ALL_LAYERS)
        self._unit_renderer = UnitRenderer()
        self._projectile_renderer = ProjectileRenderer()
        self._effect_renderer = EffectRenderer()
        self._weather_renderer = WeatherRenderer()
        self._terrain_renderer = TerrainRenderer()
        self._crowd_renderer = CrowdRenderer()

    def render_frame(self, sim_state: dict[str, Any]) -> dict[str, Any]:
        """Convert a full sim state snapshot into a Three.js frame dict.

        The *sim_state* dict may contain any subset of:
            tick, time, units, projectiles, effects, particles,
            weather, time_of_day, terrain, terrain_cell_size, cover,
            crowd, camera, ui
        """
        frame: dict[str, Any] = {
            "tick": int(sim_state.get("tick", 0)),
            "time": float(sim_state.get("time", 0.0)),
        }

        if RenderLayer.UNITS in self.layers:
            units = sim_state.get("units", [])
            frame["units"] = self._unit_renderer.render_units(units)

        if RenderLayer.PROJECTILES in self.layers:
            projectiles = sim_state.get("projectiles", [])
            frame["projectiles"] = self._projectile_renderer.render_projectiles(
                projectiles
            )

        if RenderLayer.EFFECTS in self.layers:
            effects = sim_state.get("effects", [])
            particles = sim_state.get("particles", None)
            frame["effects"] = self._effect_renderer.render_effects(
                effects, particles
            )

        if RenderLayer.WEATHER in self.layers:
            weather = sim_state.get("weather", {})
            time_of_day = sim_state.get("time_of_day", {"hour": 12.0})
            frame["weather"] = self._weather_renderer.render_weather(
                weather, time_of_day
            )

        if RenderLayer.TERRAIN in self.layers:
            heightmap = sim_state.get("terrain", None)
            if heightmap:
                cell_size = float(sim_state.get("terrain_cell_size", 1.0))
                cover = sim_state.get("cover", None)
                frame["terrain"] = self._terrain_renderer.render_terrain(
                    heightmap, cell_size, cover
                )
            else:
                frame["terrain"] = {}

        if RenderLayer.CROWD in self.layers:
            crowd = sim_state.get("crowd", [])
            frame["crowd"] = self._crowd_renderer.render_crowd(crowd)

        if RenderLayer.UI in self.layers:
            frame["ui"] = sim_state.get("ui", {})

        if RenderLayer.DEBUG in self.layers:
            frame["debug"] = sim_state.get("debug", {})

        # Camera suggestion
        camera = sim_state.get("camera", None)
        if camera:
            frame["camera"] = {
                "suggested_x": float(camera.get("x", 50.0)),
                "suggested_y": float(camera.get("y", 50.0)),
                "suggested_zoom": float(camera.get("zoom", 100.0)),
            }

        return frame

    @staticmethod
    def render_diff(
        prev_frame: dict[str, Any],
        cur_frame: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute a delta between two frames for bandwidth optimization.

        Only changed top-level keys are included.  For list-valued keys
        (units, projectiles, effects, crowd) a per-element diff by ``id``
        is performed.
        """
        diff: dict[str, Any] = {
            "tick": cur_frame.get("tick", 0),
            "time": cur_frame.get("time", 0.0),
            "is_diff": True,
        }

        list_keys = {"units", "projectiles", "effects", "crowd"}

        for key in cur_frame:
            if key in ("tick", "time"):
                continue

            prev_val = prev_frame.get(key)
            cur_val = cur_frame[key]

            if prev_val is None:
                # New key — include entirely
                diff[key] = cur_val
                continue

            if key in list_keys and isinstance(cur_val, list):
                # Per-element diff by id
                prev_by_id = {
                    item.get("id", i): item
                    for i, item in enumerate(prev_val)
                    if isinstance(item, dict)
                }
                changed: list[dict[str, Any]] = []
                removed: list[str] = []

                cur_ids = set()
                for item in cur_val:
                    if not isinstance(item, dict):
                        continue
                    item_id = item.get("id", "")
                    cur_ids.add(item_id)
                    prev_item = prev_by_id.get(item_id)
                    if prev_item != item:
                        changed.append(item)

                for pid in prev_by_id:
                    if pid not in cur_ids:
                        removed.append(str(pid))

                if changed or removed:
                    diff[key] = {"changed": changed, "removed": removed}
            else:
                if cur_val != prev_val:
                    diff[key] = cur_val

        return diff

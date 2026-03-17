# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the Three.js render bridge (sim_engine.renderer)."""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.renderer import (
    CrowdRenderer,
    EffectRenderer,
    ProjectileRenderer,
    RenderLayer,
    SimRenderer,
    TerrainRenderer,
    UnitRenderer,
    WeatherRenderer,
    alliance_color,
    damage_flash,
    mood_color,
    tracer_color,
)


# -----------------------------------------------------------------------
# Color utilities
# -----------------------------------------------------------------------


class TestAllianceColor:
    def test_friendly(self):
        assert alliance_color("friendly") == "#05ffa1"

    def test_hostile(self):
        assert alliance_color("hostile") == "#ff2a6d"

    def test_neutral(self):
        assert alliance_color("neutral") == "#00f0ff"

    def test_unknown(self):
        assert alliance_color("unknown") == "#fcee0a"

    def test_case_insensitive(self):
        assert alliance_color("Friendly") == "#05ffa1"
        assert alliance_color("HOSTILE") == "#ff2a6d"

    def test_invalid_defaults_to_unknown(self):
        assert alliance_color("banana") == "#fcee0a"


class TestMoodColor:
    def test_calm(self):
        assert mood_color("calm") == "#05ffa1"

    def test_agitated(self):
        assert mood_color("agitated") == "#fcee0a"

    def test_rioting(self):
        assert mood_color("rioting") == "#ff2a6d"

    def test_panicked(self):
        assert mood_color("panicked") == "#ff6600"

    def test_fleeing(self):
        assert mood_color("fleeing") == "#888888"

    def test_invalid_defaults_to_calm(self):
        assert mood_color("confused") == "#05ffa1"

    def test_case_insensitive(self):
        assert mood_color("CALM") == "#05ffa1"


class TestDamageFlash:
    def test_zero_is_white(self):
        assert damage_flash(0.0) == "#ffffff"

    def test_one_is_red(self):
        assert damage_flash(1.0) == "#ff0000"

    def test_half_is_midpoint(self):
        color = damage_flash(0.5)
        # green and blue should be ~127
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        assert 120 <= g <= 135
        assert 120 <= b <= 135

    def test_clamped_above_one(self):
        assert damage_flash(5.0) == "#ff0000"

    def test_clamped_below_zero(self):
        assert damage_flash(-1.0) == "#ffffff"

    def test_returns_valid_hex(self):
        color = damage_flash(0.3)
        assert color.startswith("#")
        assert len(color) == 7


class TestTracerColor:
    def test_bullet(self):
        assert tracer_color("bullet") == "#ffaa00"

    def test_laser(self):
        assert tracer_color("laser") == "#ff0000"

    def test_plasma(self):
        assert tracer_color("plasma") == "#00ccff"

    def test_missile(self):
        assert tracer_color("missile") == "#ff6600"

    def test_unknown_defaults_to_bullet(self):
        assert tracer_color("railgun") == "#ffaa00"

    def test_case_insensitive(self):
        assert tracer_color("LASER") == "#ff0000"


# -----------------------------------------------------------------------
# Unit renderer
# -----------------------------------------------------------------------


class TestUnitRenderer:
    def _make_unit(self, **overrides):
        base = {
            "id": "u1",
            "x": 10.0,
            "y": 5.0,
            "type": "infantry",
            "alliance": "friendly",
        }
        base.update(overrides)
        return base

    def test_basic_render(self):
        units = [self._make_unit()]
        result = UnitRenderer.render_units(units)
        assert len(result) == 1
        r = result[0]
        assert r["id"] == "u1"
        assert r["x"] == 10.0
        assert r["y"] == 5.0

    def test_required_fields_present(self):
        result = UnitRenderer.render_units([self._make_unit()])[0]
        required = {
            "id", "x", "y", "z", "heading", "type", "alliance",
            "color", "health", "status", "scale", "effects", "label",
            "marker_type",
        }
        assert required.issubset(set(result.keys()))

    def test_color_matches_alliance(self):
        r = UnitRenderer.render_units([self._make_unit(alliance="hostile")])[0]
        assert r["color"] == "#ff2a6d"

    def test_health_ratio(self):
        r = UnitRenderer.render_units([self._make_unit(health=50, max_health=100)])[0]
        assert r["health"] == 0.5

    def test_default_health_is_full(self):
        r = UnitRenderer.render_units([self._make_unit()])[0]
        assert r["health"] == 1.0

    def test_effects_list(self):
        r = UnitRenderer.render_units([
            self._make_unit(effects=["muzzle_flash", "smoke"])
        ])[0]
        assert r["effects"] == ["muzzle_flash", "smoke"]

    def test_marker_type_infantry(self):
        r = UnitRenderer.render_units([self._make_unit(type="infantry")])[0]
        assert r["marker_type"] == "nato_infantry"

    def test_marker_type_vehicle(self):
        r = UnitRenderer.render_units([self._make_unit(type="vehicle")])[0]
        assert r["marker_type"] == "nato_vehicle"

    def test_empty_units_list(self):
        assert UnitRenderer.render_units([]) == []

    def test_multiple_units(self):
        units = [self._make_unit(id="u1"), self._make_unit(id="u2")]
        result = UnitRenderer.render_units(units)
        assert len(result) == 2
        assert result[0]["id"] == "u1"
        assert result[1]["id"] == "u2"

    def test_label_defaults_to_id(self):
        r = UnitRenderer.render_units([self._make_unit(id="alpha")])[0]
        assert r["label"] == "alpha"

    def test_label_override(self):
        r = UnitRenderer.render_units([self._make_unit(label="Alpha-1")])[0]
        assert r["label"] == "Alpha-1"

    def test_z_defaults_to_zero(self):
        r = UnitRenderer.render_units([self._make_unit()])[0]
        assert r["z"] == 0.0


# -----------------------------------------------------------------------
# Projectile renderer
# -----------------------------------------------------------------------


class TestProjectileRenderer:
    def _make_proj(self, **overrides):
        base = {"id": "p1", "x": 25.0, "y": 12.0, "type": "bullet"}
        base.update(overrides)
        return base

    def test_basic_render(self):
        result = ProjectileRenderer.render_projectiles([self._make_proj()])
        assert len(result) == 1
        assert result[0]["id"] == "p1"

    def test_trail_length(self):
        r = ProjectileRenderer.render_projectiles([self._make_proj()])[0]
        assert "trail_length" in r
        assert isinstance(r["trail_length"], float)

    def test_glow_default_true(self):
        r = ProjectileRenderer.render_projectiles([self._make_proj()])[0]
        assert r["glow"] is True

    def test_color_by_type(self):
        r = ProjectileRenderer.render_projectiles([self._make_proj(type="laser")])[0]
        assert r["color"] == "#ff0000"

    def test_velocity_fields(self):
        r = ProjectileRenderer.render_projectiles([
            self._make_proj(vx=300, vy=10, vz=-1)
        ])[0]
        assert r["vx"] == 300.0
        assert r["vy"] == 10.0
        assert r["vz"] == -1.0

    def test_empty_list(self):
        assert ProjectileRenderer.render_projectiles([]) == []

    def test_custom_trail_length(self):
        r = ProjectileRenderer.render_projectiles([
            self._make_proj(trail_length=10.0)
        ])[0]
        assert r["trail_length"] == 10.0


# -----------------------------------------------------------------------
# Effect renderer
# -----------------------------------------------------------------------


class TestEffectRenderer:
    def test_explosion(self):
        r = EffectRenderer.render_effects([
            {"type": "explosion", "x": 30, "y": 20}
        ])[0]
        assert r["type"] == "explosion"
        assert r["radius"] == 5.0
        assert r["color"] == "#ff4400"
        assert r["particle_count"] == 50

    def test_smoke(self):
        r = EffectRenderer.render_effects([{"type": "smoke", "x": 10, "y": 15}])[0]
        assert r["type"] == "smoke"
        assert r["billboards"] is True
        assert r["opacity"] == 0.6

    def test_fire_has_emitter(self):
        r = EffectRenderer.render_effects([{"type": "fire", "x": 5, "y": 5}])[0]
        assert r["type"] == "fire"
        assert "emitter" in r
        assert r["emitter"]["rate"] == 100
        assert r["emitter"]["lifetime"] == 1.5

    def test_flashbang(self):
        r = EffectRenderer.render_effects([{"type": "flashbang", "x": 0, "y": 0}])[0]
        assert r["type"] == "flashbang"
        assert r["color"] == "#ffffff"
        assert r["screen_flash"] is True

    def test_age_field(self):
        r = EffectRenderer.render_effects([
            {"type": "explosion", "x": 0, "y": 0, "age": 0.3}
        ])[0]
        assert r["age"] == 0.3

    def test_age_defaults_to_zero(self):
        r = EffectRenderer.render_effects([{"type": "explosion", "x": 0, "y": 0}])[0]
        assert r["age"] == 0.0

    def test_override_defaults(self):
        r = EffectRenderer.render_effects([
            {"type": "explosion", "x": 0, "y": 0, "radius": 20.0, "color": "#00ff00"}
        ])[0]
        assert r["radius"] == 20.0
        assert r["color"] == "#00ff00"

    def test_particles_appended(self):
        result = EffectRenderer.render_effects(
            [{"type": "smoke", "x": 0, "y": 0}],
            particles=[{"x": 1, "y": 2, "color": "#ff0000", "size": 2.0}],
        )
        assert len(result) == 2
        p = result[1]
        assert p["type"] == "particle"
        assert p["x"] == 1.0
        assert p["color"] == "#ff0000"

    def test_empty_effects(self):
        assert EffectRenderer.render_effects([]) == []

    def test_empty_effects_with_particles(self):
        result = EffectRenderer.render_effects(
            [], particles=[{"x": 0, "y": 0}]
        )
        assert len(result) == 1
        assert result[0]["type"] == "particle"


# -----------------------------------------------------------------------
# Weather renderer
# -----------------------------------------------------------------------


class TestWeatherRenderer:
    def test_daytime_bright(self):
        r = WeatherRenderer.render_weather({}, {"hour": 12.0})
        assert r["ambient_light"] >= 0.7

    def test_nighttime_dark(self):
        r = WeatherRenderer.render_weather({}, {"hour": 0.0})
        assert r["ambient_light"] <= 0.2

    def test_rain_particles(self):
        r = WeatherRenderer.render_weather({"rain": 0.7}, {"hour": 12.0})
        assert "rain" in r
        assert r["rain"]["intensity"] == 0.7
        assert r["rain"]["color"] == "#aaccff"
        assert "particles" in r
        assert r["particles"]["type"] == "rain"
        assert r["particles"]["count"] > 0

    def test_snow_particles(self):
        r = WeatherRenderer.render_weather({"snow": 0.5}, {"hour": 12.0})
        assert "snow" in r
        assert r["particles"]["type"] == "snow"
        assert r["particles"]["count"] > 0

    def test_fog_density(self):
        r = WeatherRenderer.render_weather({"fog": 0.05}, {"hour": 12.0})
        assert r["fog_density"] == 0.05

    def test_rain_increases_fog(self):
        no_rain = WeatherRenderer.render_weather({"fog": 0.01}, {"hour": 12.0})
        with_rain = WeatherRenderer.render_weather(
            {"fog": 0.01, "rain": 0.5}, {"hour": 12.0}
        )
        assert with_rain["fog_density"] > no_rain["fog_density"]

    def test_sun_angle_noon(self):
        r = WeatherRenderer.render_weather({}, {"hour": 12.0})
        assert r["sun"]["angle"] > 80  # near 90 at noon

    def test_sun_angle_midnight(self):
        r = WeatherRenderer.render_weather({}, {"hour": 0.0})
        assert r["sun"]["angle"] < 0  # below horizon

    def test_wind_present(self):
        r = WeatherRenderer.render_weather(
            {"wind_speed": 5.0, "wind_direction": 1.5}, {"hour": 12.0}
        )
        assert r["wind"]["speed"] == 5.0
        assert r["wind"]["direction"] == 1.5

    def test_sky_color_changes_with_time(self):
        day = WeatherRenderer.render_weather({}, {"hour": 12.0})
        night = WeatherRenderer.render_weather({}, {"hour": 0.0})
        assert day["sky_color"] != night["sky_color"]

    def test_overcast_darkens(self):
        clear = WeatherRenderer.render_weather({}, {"hour": 12.0})
        overcast = WeatherRenderer.render_weather({"overcast": 0.8}, {"hour": 12.0})
        assert overcast["ambient_light"] < clear["ambient_light"]

    def test_no_rain_no_rain_key(self):
        r = WeatherRenderer.render_weather({}, {"hour": 12.0})
        assert "rain" not in r


# -----------------------------------------------------------------------
# Terrain renderer
# -----------------------------------------------------------------------


class TestTerrainRenderer:
    def _simple_heightmap(self, rows=3, cols=3):
        return [[float(r + c) for c in range(cols)] for r in range(rows)]

    def test_vertex_count(self):
        hm = self._simple_heightmap(4, 5)
        r = TerrainRenderer.render_terrain(hm, 1.0)
        # 4*5 = 20 vertices, 3 floats each
        assert len(r["vertices"]) == 4 * 5 * 3

    def test_face_count(self):
        hm = self._simple_heightmap(4, 5)
        r = TerrainRenderer.render_terrain(hm, 1.0)
        # (4-1)*(5-1)*2 triangles, 3 indices each
        expected_faces = 3 * 4 * 2 * 3
        assert len(r["faces"]) == expected_faces

    def test_color_per_vertex(self):
        hm = self._simple_heightmap(3, 3)
        r = TerrainRenderer.render_terrain(hm, 1.0)
        assert len(r["colors"]) == 9

    def test_colors_are_hex(self):
        hm = self._simple_heightmap(2, 2)
        r = TerrainRenderer.render_terrain(hm, 1.0)
        for c in r["colors"]:
            assert c.startswith("#")
            assert len(c) == 7

    def test_empty_heightmap(self):
        r = TerrainRenderer.render_terrain([], 1.0)
        assert r["vertices"] == []
        assert r["faces"] == []

    def test_cell_size_scales_positions(self):
        hm = [[0.0, 1.0], [2.0, 3.0]]
        r = TerrainRenderer.render_terrain(hm, 2.0)
        # Second vertex should be at x=2.0 (1 * cell_size)
        assert r["vertices"][3] == 2.0  # x of vertex[1]

    def test_cover_overlay(self):
        hm = self._simple_heightmap(2, 2)
        cover = {"cells": {"5,3": 0.8, "1,2": 0.3}}
        r = TerrainRenderer.render_terrain(hm, 1.0, cover_data=cover)
        assert len(r["cover_overlay"]) == 2
        overlay_vals = {o["value"] for o in r["cover_overlay"]}
        assert 0.8 in overlay_vals

    def test_no_cover_gives_empty_overlay(self):
        hm = self._simple_heightmap(2, 2)
        r = TerrainRenderer.render_terrain(hm, 1.0)
        assert r["cover_overlay"] == []


class TestLOSOverlay:
    def test_basic_fog(self):
        visible = {(0, 0), (1, 1), (2, 0)}
        r = TerrainRenderer.render_los_overlay(visible, (3, 3))
        assert r["width"] == 3
        assert r["height"] == 3
        assert r["fog"][0][0] == 1  # (0,0) visible
        assert r["fog"][0][2] == 1  # (2,0) is visible
        assert r["fog"][1][1] == 1  # (1,1) visible

    def test_all_visible(self):
        visible = {(c, r) for r in range(2) for c in range(2)}
        r = TerrainRenderer.render_los_overlay(visible, (2, 2))
        for row in r["fog"]:
            assert all(v == 1 for v in row)

    def test_none_visible(self):
        r = TerrainRenderer.render_los_overlay(set(), (3, 3))
        for row in r["fog"]:
            assert all(v == 0 for v in row)

    def test_fog_color_present(self):
        r = TerrainRenderer.render_los_overlay(set(), (2, 2))
        assert "fog_color" in r
        assert "fog_opacity" in r


# -----------------------------------------------------------------------
# SimRenderer (combined)
# -----------------------------------------------------------------------


class TestSimRenderer:
    def _make_sim_state(self):
        return {
            "tick": 42,
            "time": 4.2,
            "units": [
                {"id": "u1", "x": 10, "y": 5, "type": "infantry", "alliance": "friendly"},
            ],
            "projectiles": [
                {"id": "p1", "x": 25, "y": 12, "type": "bullet"},
            ],
            "effects": [
                {"type": "explosion", "x": 30, "y": 20},
            ],
            "weather": {"rain": 0.3},
            "time_of_day": {"hour": 14.0},
            "terrain": [[0.0, 1.0], [2.0, 3.0]],
            "terrain_cell_size": 1.0,
            "crowd": [
                {"id": "c1", "x": 5, "y": 5, "mood": "calm"},
            ],
            "camera": {"x": 50, "y": 50, "zoom": 100},
            "ui": {"wave": 3, "score": 1500, "phase": "active"},
        }

    def test_full_frame(self):
        renderer = SimRenderer()
        frame = renderer.render_frame(self._make_sim_state())
        assert frame["tick"] == 42
        assert frame["time"] == 4.2
        assert "units" in frame
        assert "projectiles" in frame
        assert "effects" in frame
        assert "weather" in frame
        assert "terrain" in frame
        assert "crowd" in frame
        assert "ui" in frame
        assert "camera" in frame

    def test_layer_filtering(self):
        renderer = SimRenderer(layers={RenderLayer.UNITS})
        frame = renderer.render_frame(self._make_sim_state())
        assert "units" in frame
        assert "projectiles" not in frame
        assert "effects" not in frame

    def test_empty_state(self):
        renderer = SimRenderer()
        frame = renderer.render_frame({})
        assert frame["tick"] == 0
        assert frame["time"] == 0.0
        assert frame["units"] == []

    def test_camera_suggestion(self):
        renderer = SimRenderer()
        frame = renderer.render_frame(self._make_sim_state())
        assert frame["camera"]["suggested_x"] == 50.0
        assert frame["camera"]["suggested_zoom"] == 100.0

    def test_no_camera_no_key(self):
        renderer = SimRenderer()
        frame = renderer.render_frame({"tick": 1})
        assert "camera" not in frame

    def test_ui_passthrough(self):
        renderer = SimRenderer()
        frame = renderer.render_frame(self._make_sim_state())
        assert frame["ui"]["wave"] == 3
        assert frame["ui"]["phase"] == "active"

    def test_terrain_empty_when_missing(self):
        renderer = SimRenderer()
        frame = renderer.render_frame({"tick": 1})
        assert frame["terrain"] == {}


class TestRenderDiff:
    def test_no_change(self):
        frame = {"tick": 1, "time": 0.1, "weather": {"sky": "blue"}}
        diff = SimRenderer.render_diff(frame, frame)
        assert diff["is_diff"] is True
        # weather unchanged, should not appear
        assert "weather" not in diff

    def test_new_key(self):
        prev = {"tick": 1, "time": 0.1}
        cur = {"tick": 2, "time": 0.2, "weather": {"sky": "red"}}
        diff = SimRenderer.render_diff(prev, cur)
        assert "weather" in diff
        assert diff["weather"]["sky"] == "red"

    def test_changed_dict(self):
        prev = {"tick": 1, "time": 0.1, "weather": {"sky": "blue"}}
        cur = {"tick": 2, "time": 0.2, "weather": {"sky": "red"}}
        diff = SimRenderer.render_diff(prev, cur)
        assert diff["weather"]["sky"] == "red"

    def test_unit_diff_changed(self):
        prev = {
            "tick": 1, "time": 0.1,
            "units": [{"id": "u1", "x": 0, "y": 0}],
        }
        cur = {
            "tick": 2, "time": 0.2,
            "units": [{"id": "u1", "x": 5, "y": 0}],
        }
        diff = SimRenderer.render_diff(prev, cur)
        assert "units" in diff
        assert len(diff["units"]["changed"]) == 1
        assert diff["units"]["changed"][0]["x"] == 5

    def test_unit_diff_removed(self):
        prev = {
            "tick": 1, "time": 0.1,
            "units": [{"id": "u1", "x": 0, "y": 0}],
        }
        cur = {"tick": 2, "time": 0.2, "units": []}
        diff = SimRenderer.render_diff(prev, cur)
        assert "u1" in diff["units"]["removed"]

    def test_unit_diff_added(self):
        prev = {"tick": 1, "time": 0.1, "units": []}
        cur = {
            "tick": 2, "time": 0.2,
            "units": [{"id": "u2", "x": 10, "y": 10}],
        }
        diff = SimRenderer.render_diff(prev, cur)
        assert len(diff["units"]["changed"]) == 1
        assert diff["units"]["changed"][0]["id"] == "u2"

    def test_diff_has_tick_and_time(self):
        diff = SimRenderer.render_diff(
            {"tick": 1, "time": 0.1},
            {"tick": 2, "time": 0.2},
        )
        assert diff["tick"] == 2
        assert diff["time"] == 0.2


# -----------------------------------------------------------------------
# RenderLayer enum
# -----------------------------------------------------------------------


class TestRenderLayer:
    def test_all_layers(self):
        assert len(RenderLayer) == 8

    def test_values(self):
        expected = {
            "units", "projectiles", "effects", "terrain",
            "crowd", "weather", "ui", "debug",
        }
        assert {layer.value for layer in RenderLayer} == expected


# -----------------------------------------------------------------------
# Crowd renderer
# -----------------------------------------------------------------------


class TestCrowdRenderer:
    def test_basic(self):
        result = CrowdRenderer.render_crowd([
            {"id": "c1", "x": 5, "y": 10, "mood": "panicked"}
        ])
        assert len(result) == 1
        assert result[0]["color"] == "#ff6600"
        assert result[0]["mood"] == "panicked"

    def test_empty(self):
        assert CrowdRenderer.render_crowd([]) == []

    def test_default_mood(self):
        result = CrowdRenderer.render_crowd([{"id": "c1", "x": 0, "y": 0}])
        assert result[0]["mood"] == "calm"
        assert result[0]["color"] == "#05ffa1"

    def test_crowd_heading_and_speed(self):
        result = CrowdRenderer.render_crowd([
            {"id": "c1", "x": 5, "y": 5, "heading": 1.57, "speed": 3.0}
        ])
        assert result[0]["heading"] == 1.57
        assert result[0]["speed"] == 3.0

    def test_crowd_scale_default(self):
        result = CrowdRenderer.render_crowd([{"id": "c1", "x": 0, "y": 0}])
        assert result[0]["scale"] == 0.8

    def test_crowd_multiple_moods(self):
        crowd = [
            {"id": "c1", "x": 0, "y": 0, "mood": "calm"},
            {"id": "c2", "x": 1, "y": 1, "mood": "panicked"},
            {"id": "c3", "x": 2, "y": 2, "mood": "fleeing"},
        ]
        result = CrowdRenderer.render_crowd(crowd)
        assert len(result) == 3
        colors = {r["color"] for r in result}
        assert len(colors) == 3  # all different colors


# -----------------------------------------------------------------------
# JSON serialization safety
# -----------------------------------------------------------------------


class TestJSONSafety:
    """Verify all renderer outputs are JSON-serializable."""

    def test_unit_render_json_safe(self):
        import json
        units = [
            {"id": "u1", "x": 10, "y": 5, "type": "infantry", "alliance": "friendly"},
            {"id": "u2", "x": 20, "y": 15, "type": "sniper", "alliance": "hostile",
             "health": 75, "max_health": 100, "effects": ["smoke"]},
        ]
        result = UnitRenderer.render_units(units)
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert len(parsed) == 2

    def test_projectile_render_json_safe(self):
        import json
        projs = [
            {"id": "p1", "x": 25, "y": 12, "type": "bullet", "vx": 300, "vy": 10},
        ]
        result = ProjectileRenderer.render_projectiles(projs)
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert len(parsed) == 1

    def test_effect_render_json_safe(self):
        import json
        effects = [
            {"type": "explosion", "x": 30, "y": 20},
            {"type": "smoke", "x": 10, "y": 15},
            {"type": "fire", "x": 5, "y": 5},
        ]
        result = EffectRenderer.render_effects(effects)
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert len(parsed) == 3

    def test_weather_render_json_safe(self):
        import json
        result = WeatherRenderer.render_weather(
            {"rain": 0.5, "fog": 0.02, "wind_speed": 5.0},
            {"hour": 14.0},
        )
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert "sky_color" in parsed

    def test_terrain_render_json_safe(self):
        import json
        hm = [[0.0, 1.0], [2.0, 3.0]]
        result = TerrainRenderer.render_terrain(hm, 1.0)
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert "vertices" in parsed

    def test_full_frame_json_safe(self):
        import json
        renderer = SimRenderer()
        state = {
            "tick": 10,
            "time": 1.0,
            "units": [{"id": "u1", "x": 10, "y": 5, "type": "infantry", "alliance": "friendly"}],
            "projectiles": [{"id": "p1", "x": 25, "y": 12, "type": "bullet"}],
            "effects": [{"type": "explosion", "x": 30, "y": 20}],
            "weather": {"rain": 0.3},
            "time_of_day": {"hour": 14.0},
            "crowd": [{"id": "c1", "x": 5, "y": 5, "mood": "calm"}],
        }
        frame = renderer.render_frame(state)
        serialized = json.dumps(frame)
        parsed = json.loads(serialized)
        assert parsed["tick"] == 10

    def test_diff_json_safe(self):
        import json
        prev = {
            "tick": 1, "time": 0.1,
            "units": [{"id": "u1", "x": 0, "y": 0}],
        }
        cur = {
            "tick": 2, "time": 0.2,
            "units": [{"id": "u1", "x": 5, "y": 3}],
        }
        diff = SimRenderer.render_diff(prev, cur)
        serialized = json.dumps(diff)
        parsed = json.loads(serialized)
        assert parsed["is_diff"] is True


# -----------------------------------------------------------------------
# SimRenderer layer combinations
# -----------------------------------------------------------------------


class TestSimRendererLayers:
    """Test various layer combinations for SimRenderer."""

    def test_units_only(self):
        renderer = SimRenderer(layers={RenderLayer.UNITS})
        frame = renderer.render_frame({
            "tick": 1, "time": 0.1,
            "units": [{"id": "u1", "x": 0, "y": 0, "type": "infantry", "alliance": "friendly"}],
            "projectiles": [{"id": "p1", "x": 5, "y": 5, "type": "bullet"}],
        })
        assert "units" in frame
        assert "projectiles" not in frame

    def test_weather_only(self):
        renderer = SimRenderer(layers={RenderLayer.WEATHER})
        frame = renderer.render_frame({
            "tick": 1, "time": 0.1,
            "weather": {"rain": 0.5},
            "time_of_day": {"hour": 12.0},
        })
        assert "weather" in frame
        assert "units" not in frame

    def test_debug_layer(self):
        renderer = SimRenderer(layers={RenderLayer.DEBUG})
        frame = renderer.render_frame({
            "tick": 1, "time": 0.1,
            "debug": {"pathfinding": True, "los_rays": 42},
        })
        assert frame["debug"]["pathfinding"] is True
        assert frame["debug"]["los_rays"] == 42

    def test_multiple_layers(self):
        renderer = SimRenderer(layers={RenderLayer.UNITS, RenderLayer.EFFECTS, RenderLayer.CROWD})
        frame = renderer.render_frame({
            "tick": 1, "time": 0.1,
            "units": [{"id": "u1", "x": 0, "y": 0, "type": "infantry", "alliance": "friendly"}],
            "effects": [{"type": "smoke", "x": 10, "y": 10}],
            "crowd": [{"id": "c1", "x": 5, "y": 5, "mood": "calm"}],
            "weather": {"rain": 0.3},
            "time_of_day": {"hour": 12.0},
        })
        assert "units" in frame
        assert "effects" in frame
        assert "crowd" in frame
        assert "weather" not in frame

    def test_all_layers_is_default(self):
        renderer = SimRenderer()
        assert renderer.layers == set(RenderLayer)

    def test_terrain_with_no_heightmap(self):
        renderer = SimRenderer(layers={RenderLayer.TERRAIN})
        frame = renderer.render_frame({"tick": 1, "time": 0.1})
        assert frame["terrain"] == {}


# -----------------------------------------------------------------------
# Weather edge cases
# -----------------------------------------------------------------------


class TestWeatherEdgeCases:
    def test_dawn_transition(self):
        """Hour 5.5 should produce a twilight sky."""
        r = WeatherRenderer.render_weather({}, {"hour": 5.5})
        assert 0.1 <= r["ambient_light"] <= 0.5

    def test_dusk_transition(self):
        """Hour 19.0 should produce dimming light."""
        r = WeatherRenderer.render_weather({}, {"hour": 19.0})
        assert r["ambient_light"] < 0.8

    def test_overcast_changes_sky(self):
        clear = WeatherRenderer.render_weather({}, {"hour": 12.0})
        overcast = WeatherRenderer.render_weather({"overcast": 1.0}, {"hour": 12.0})
        assert clear["sky_color"] != overcast["sky_color"]

    def test_sun_intensity_reduced_by_overcast(self):
        clear = WeatherRenderer.render_weather({}, {"hour": 12.0})
        overcast = WeatherRenderer.render_weather({"overcast": 0.9}, {"hour": 12.0})
        assert overcast["sun"]["intensity"] < clear["sun"]["intensity"]

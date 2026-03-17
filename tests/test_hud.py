# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for sim_engine HUD system — minimap, compass, roster, resources,
notifications, kill feed, and the HUDEngine compositor.

Run::

    cd /home/scubasonar/Code/tritium
    python3 -m pytest tritium-lib/tests/test_hud.py -v --tb=short

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import math
import time

import pytest

from tritium_lib.sim_engine.hud import (
    CompassHUD,
    HUDEngine,
    KillFeed,
    MinimapRenderer,
    NotificationPriority,
    NotificationQueue,
    ResourceHUD,
    UnitRoster,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit(
    uid: str = "u1",
    pos: tuple[float, float] = (100.0, 100.0),
    alliance: str = "friendly",
    health: float = 80.0,
    max_health: float = 100.0,
    ammo: int = 30,
    status: str = "idle",
    weapon: str = "rifle",
    is_alive: bool = True,
    distance: float = 0.0,
) -> dict:
    return {
        "unit_id": uid,
        "name": f"Unit-{uid}",
        "position": pos,
        "alliance": alliance,
        "health": health,
        "max_health": max_health,
        "ammo": ammo,
        "status": status,
        "weapon": weapon,
        "is_alive": is_alive,
        "distance": distance,
    }


def _vehicle(
    vid: str = "v1",
    pos: tuple[float, float] = (200.0, 200.0),
    alliance: str = "friendly",
    is_destroyed: bool = False,
) -> dict:
    return {
        "vehicle_id": vid,
        "position": pos,
        "alliance": alliance,
        "is_destroyed": is_destroyed,
    }


def _structure(
    bid: str = "b1",
    pos: tuple[float, float] = (300.0, 300.0),
    alliance: str = "neutral",
) -> dict:
    return {
        "building_id": bid,
        "position": pos,
        "alliance": alliance,
    }


# ===========================================================================
# MinimapRenderer
# ===========================================================================


class TestMinimapRenderer:
    """Tests for MinimapRenderer."""

    def test_creation(self):
        mm = MinimapRenderer(500.0, 500.0, 200)
        assert mm.map_width == 500.0
        assert mm.minimap_size == 200

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            MinimapRenderer(0, 500)
        with pytest.raises(ValueError):
            MinimapRenderer(500, -1)
        with pytest.raises(ValueError):
            MinimapRenderer(500, 500, 0)

    def test_world_to_minimap_origin(self):
        mm = MinimapRenderer(500.0, 500.0, 200)
        px, py = mm.world_to_minimap((0.0, 0.0))
        assert px == 0.0
        assert py == 0.0

    def test_world_to_minimap_center(self):
        mm = MinimapRenderer(500.0, 500.0, 200)
        px, py = mm.world_to_minimap((250.0, 250.0))
        assert px == 100.0
        assert py == 100.0

    def test_world_to_minimap_max(self):
        mm = MinimapRenderer(500.0, 500.0, 200)
        px, py = mm.world_to_minimap((500.0, 500.0))
        assert px == 200.0
        assert py == 200.0

    def test_world_to_minimap_clamps(self):
        mm = MinimapRenderer(500.0, 500.0, 200)
        px, py = mm.world_to_minimap((-50.0, 600.0))
        assert px == 0.0
        assert py == 200.0

    def test_render_empty(self):
        mm = MinimapRenderer(500.0, 500.0)
        result = mm.render(units=[])
        assert result["marker_count"] == 0
        assert result["markers"] == []
        assert result["minimap_size"] == 200

    def test_render_units(self):
        mm = MinimapRenderer(500.0, 500.0, 200)
        units = [_unit("u1", (100, 100), "friendly"), _unit("u2", (400, 400), "hostile")]
        result = mm.render(units=units)
        assert result["marker_count"] == 2
        assert result["markers"][0]["shape"] == "dot"
        assert result["markers"][0]["color"] == "#05ffa1"  # friendly green
        assert result["markers"][1]["color"] == "#ff2a6d"  # hostile red

    def test_render_dead_units_excluded(self):
        mm = MinimapRenderer(500.0, 500.0)
        units = [_unit("u1", is_alive=False)]
        result = mm.render(units=units)
        assert result["marker_count"] == 0

    def test_render_vehicles(self):
        mm = MinimapRenderer(500.0, 500.0)
        result = mm.render(units=[], vehicles=[_vehicle("v1")])
        assert result["marker_count"] == 1
        assert result["markers"][0]["shape"] == "square"

    def test_render_destroyed_vehicles_excluded(self):
        mm = MinimapRenderer(500.0, 500.0)
        result = mm.render(units=[], vehicles=[_vehicle("v1", is_destroyed=True)])
        assert result["marker_count"] == 0

    def test_render_structures(self):
        mm = MinimapRenderer(500.0, 500.0)
        result = mm.render(units=[], structures=[_structure("b1")])
        assert result["marker_count"] == 1
        assert result["markers"][0]["shape"] == "rect"

    def test_render_fog_of_war(self):
        mm = MinimapRenderer(500.0, 500.0)
        fog = {(0, 0), (1, 1), (2, 2)}
        result = mm.render(units=[], fog_of_war=fog)
        assert len(result["fog_cells"]) == 3

    def test_render_mixed_entities(self):
        mm = MinimapRenderer(1000.0, 1000.0)
        result = mm.render(
            units=[_unit("u1"), _unit("u2")],
            vehicles=[_vehicle("v1")],
            structures=[_structure("b1")],
        )
        assert result["marker_count"] == 4

    def test_to_three_js(self):
        mm = MinimapRenderer(500.0, 500.0, 200)
        result = mm.to_three_js(camera_pos=(250.0, 250.0), camera_fov=60.0)
        assert "camera" in result
        assert result["camera"]["x"] == 100.0
        assert "viewport" in result
        assert result["viewport"]["width"] > 0

    def test_to_three_js_with_markers(self):
        mm = MinimapRenderer(500.0, 500.0, 200)
        markers = [{"x": 10, "y": 10, "shape": "dot"}]
        result = mm.to_three_js((0, 0), 60.0, markers=markers)
        assert len(result["markers"]) == 1

    def test_non_square_map(self):
        mm = MinimapRenderer(1000.0, 500.0, 200)
        px, py = mm.world_to_minimap((500.0, 250.0))
        assert px == 100.0
        assert py == 100.0


# ===========================================================================
# CompassHUD
# ===========================================================================


class TestCompassHUD:
    """Tests for CompassHUD."""

    def test_bearing_north(self):
        """Target directly north (positive y) should give 0 degrees."""
        bearing = CompassHUD.get_bearing((0, 0), (0, 10))
        assert abs(bearing - 0.0) < 0.01

    def test_bearing_east(self):
        """Target directly east (positive x) should give 90 degrees."""
        bearing = CompassHUD.get_bearing((0, 0), (10, 0))
        assert abs(bearing - 90.0) < 0.01

    def test_bearing_south(self):
        bearing = CompassHUD.get_bearing((0, 0), (0, -10))
        assert abs(bearing - 180.0) < 0.01

    def test_bearing_west(self):
        bearing = CompassHUD.get_bearing((0, 0), (-10, 0))
        assert abs(bearing - 270.0) < 0.01

    def test_bearing_northeast(self):
        bearing = CompassHUD.get_bearing((0, 0), (10, 10))
        assert abs(bearing - 45.0) < 0.01

    def test_cardinal_n(self):
        assert CompassHUD.get_cardinal(0.0) == "N"
        assert CompassHUD.get_cardinal(360.0) == "N"

    def test_cardinal_e(self):
        assert CompassHUD.get_cardinal(90.0) == "E"

    def test_cardinal_s(self):
        assert CompassHUD.get_cardinal(180.0) == "S"

    def test_cardinal_w(self):
        assert CompassHUD.get_cardinal(270.0) == "W"

    def test_cardinal_ne(self):
        assert CompassHUD.get_cardinal(45.0) == "NE"

    def test_cardinal_se(self):
        assert CompassHUD.get_cardinal(135.0) == "SE"

    def test_cardinal_sw(self):
        assert CompassHUD.get_cardinal(225.0) == "SW"

    def test_cardinal_nw(self):
        assert CompassHUD.get_cardinal(315.0) == "NW"

    def test_cardinal_wraps(self):
        assert CompassHUD.get_cardinal(720.0) == "N"

    def test_render_empty(self):
        c = CompassHUD()
        result = c.render(player_heading=0.0)
        assert result["heading"] == 0.0
        assert result["cardinal"] == "N"
        assert result["threat_count"] == 0
        assert result["objective_count"] == 0

    def test_render_with_threats(self):
        c = CompassHUD()
        threats = [
            {"position": (100, 0), "player_pos": (0, 0), "label": "Sniper"},
        ]
        result = c.render(player_heading=0.0, threats=threats)
        assert result["threat_count"] == 1
        assert result["threats"][0]["color"] == "#ff2a6d"
        assert result["threats"][0]["label"] == "Sniper"

    def test_render_with_objectives(self):
        c = CompassHUD()
        objectives = [
            {"position": (0, 100), "player_pos": (0, 0), "label": "Alpha"},
        ]
        result = c.render(player_heading=0.0, objectives=objectives)
        assert result["objective_count"] == 1
        assert result["objectives"][0]["shape"] == "diamond"

    def test_relative_bearing(self):
        c = CompassHUD()
        threats = [{"position": (10, 0), "player_pos": (0, 0)}]
        result = c.render(player_heading=90.0, threats=threats)
        # Target is at bearing 90, player heading 90 -> relative ~0
        rel = result["threats"][0]["relative_bearing"]
        assert abs(rel) < 1.0 or abs(rel - 360.0) < 1.0

    def test_heading_normalization(self):
        c = CompassHUD()
        result = c.render(player_heading=450.0)
        assert result["heading"] == 90.0


# ===========================================================================
# UnitRoster
# ===========================================================================


class TestUnitRoster:
    """Tests for UnitRoster."""

    def test_render_empty(self):
        r = UnitRoster()
        result = r.render(units=[])
        assert result["count"] == 0

    def test_render_filters_alliance(self):
        r = UnitRoster()
        units = [_unit("u1", alliance="friendly"), _unit("u2", alliance="hostile")]
        result = r.render(units, alliance="friendly")
        assert result["count"] == 1
        assert result["entries"][0]["unit_id"] == "u1"

    def test_render_excludes_dead(self):
        r = UnitRoster()
        units = [_unit("u1", is_alive=False, alliance="friendly")]
        result = r.render(units, alliance="friendly")
        assert result["count"] == 0

    def test_health_bar_colors(self):
        r = UnitRoster()
        units = [
            _unit("u1", health=80, max_health=100, alliance="friendly"),  # green
            _unit("u2", health=40, max_health=100, alliance="friendly"),  # yellow
            _unit("u3", health=10, max_health=100, alliance="friendly"),  # red
        ]
        result = r.render(units, alliance="friendly")
        colors = [e["health_bar_color"] for e in result["entries"]]
        # Sorted by health ascending: u3(10%), u2(40%), u1(80%)
        assert colors[0] == "#ff2a6d"    # 10% red
        assert colors[1] == "#fcee0a"    # 40% yellow
        assert colors[2] == "#05ffa1"    # 80% green

    def test_ammo_display_unlimited(self):
        r = UnitRoster()
        units = [_unit("u1", ammo=-1, alliance="friendly")]
        result = r.render(units, alliance="friendly")
        assert result["entries"][0]["ammo_display"] == "INF"

    def test_ammo_display_number(self):
        r = UnitRoster()
        units = [_unit("u1", ammo=30, alliance="friendly")]
        result = r.render(units, alliance="friendly")
        assert result["entries"][0]["ammo_display"] == "30"

    def test_sort_by_health(self):
        r = UnitRoster()
        units = [
            _unit("u1", health=80, alliance="friendly"),
            _unit("u2", health=20, alliance="friendly"),
            _unit("u3", health=50, alliance="friendly"),
        ]
        result = r.render(units, alliance="friendly", sort_by="health")
        ids = [e["unit_id"] for e in result["entries"]]
        assert ids == ["u2", "u3", "u1"]

    def test_sort_by_distance(self):
        r = UnitRoster()
        units = [
            _unit("u1", distance=100.0, alliance="friendly"),
            _unit("u2", distance=10.0, alliance="friendly"),
            _unit("u3", distance=50.0, alliance="friendly"),
        ]
        result = r.render(units, alliance="friendly", sort_by="distance")
        ids = [e["unit_id"] for e in result["entries"]]
        assert ids == ["u2", "u3", "u1"]

    def test_sort_by_status(self):
        r = UnitRoster()
        units = [
            _unit("u1", status="idle", alliance="friendly"),
            _unit("u2", status="attacking", alliance="friendly"),
            _unit("u3", status="moving", alliance="friendly"),
        ]
        result = r.render(units, alliance="friendly", sort_by="status")
        statuses = [e["status"] for e in result["entries"]]
        assert statuses == ["attacking", "moving", "idle"]

    def test_health_pct_zero_max(self):
        r = UnitRoster()
        units = [_unit("u1", health=50, max_health=0, alliance="friendly")]
        result = r.render(units, alliance="friendly")
        assert result["entries"][0]["health_pct"] == 0.0


# ===========================================================================
# ResourceHUD
# ===========================================================================


class TestResourceHUD:
    """Tests for ResourceHUD."""

    def test_render_empty(self):
        rh = ResourceHUD()
        result = rh.render()
        assert result["bar_count"] == 0

    def test_render_single_resource(self):
        rh = ResourceHUD()
        result = rh.render(economy_state={"ammo": 50.0}, capacity={"ammo": 100.0})
        assert result["bar_count"] == 1
        bar = result["bars"][0]
        assert bar["name"] == "ammo"
        assert bar["pct"] == 50.0
        assert bar["bar_color"] == "#05ffa1"  # 50% -> green

    def test_low_warning(self):
        rh = ResourceHUD()
        result = rh.render(economy_state={"fuel": 15.0}, capacity={"fuel": 100.0})
        assert result["warning_count"] == 1
        assert result["warnings"][0]["level"] == "low"

    def test_critical_warning(self):
        rh = ResourceHUD()
        result = rh.render(economy_state={"fuel": 5.0}, capacity={"fuel": 100.0})
        assert result["warning_count"] == 1
        assert result["warnings"][0]["level"] == "critical"

    def test_no_warning_above_threshold(self):
        rh = ResourceHUD()
        result = rh.render(economy_state={"ammo": 80.0}, capacity={"ammo": 100.0})
        assert result["warning_count"] == 0

    def test_income_expense_flow(self):
        rh = ResourceHUD()
        result = rh.render(
            economy_state={"credits": 500.0},
            capacity={"credits": 1000.0},
            income={"credits": 10.0},
            expenses={"credits": 3.0},
        )
        bar = result["bars"][0]
        assert bar["net"] == 7.0
        assert bar["flow_icon"] == "arrow_up"
        assert bar["flow_color"] == "#05ffa1"

    def test_negative_flow(self):
        rh = ResourceHUD()
        result = rh.render(
            economy_state={"fuel": 50.0},
            capacity={"fuel": 100.0},
            income={"fuel": 1.0},
            expenses={"fuel": 5.0},
        )
        bar = result["bars"][0]
        assert bar["net"] == -4.0
        assert bar["flow_icon"] == "arrow_down"

    def test_supply_merged(self):
        rh = ResourceHUD()
        result = rh.render(
            economy_state={"credits": 100.0},
            supply_state={"ammo": 50.0},
        )
        assert result["bar_count"] == 2

    def test_resource_icon(self):
        rh = ResourceHUD()
        result = rh.render(economy_state={"medical": 30.0}, capacity={"medical": 100.0})
        assert result["bars"][0]["icon"] == "medkit"

    def test_no_capacity_defaults_100pct(self):
        rh = ResourceHUD()
        result = rh.render(economy_state={"ammo": 50.0})
        # No capacity entry -> pct = 100
        assert result["bars"][0]["pct"] == 100.0
        assert result["warning_count"] == 0


# ===========================================================================
# NotificationQueue
# ===========================================================================


class TestNotificationQueue:
    """Tests for NotificationQueue."""

    def test_add_and_tick(self):
        nq = NotificationQueue()
        nq.add("Hello", priority="medium", duration=5.0)
        result = nq.tick(0.0)
        assert len(result) == 1
        assert result[0]["text"] == "Hello"

    def test_expiry(self):
        nq = NotificationQueue()
        nq.add("Short", duration=2.0)
        nq.tick(1.0)
        result = nq.tick(1.5)  # age = 2.5, past 2.0 duration
        assert len(result) == 0

    def test_priority_ordering(self):
        nq = NotificationQueue()
        nq.add("Low", priority="low", duration=10.0)
        nq.add("Critical", priority="critical", duration=10.0)
        nq.add("High", priority="high", duration=10.0)
        result = nq.tick(0.0)
        priorities = [r["priority"] for r in result]
        assert priorities[0] == "critical"
        assert priorities[1] == "high"
        assert priorities[2] == "low"

    def test_max_visible(self):
        nq = NotificationQueue(max_visible=2)
        for i in range(5):
            nq.add(f"Notif-{i}", duration=10.0)
        result = nq.tick(0.0)
        assert len(result) == 2

    def test_dismiss(self):
        nq = NotificationQueue()
        nid = nq.add("Dismiss me", duration=10.0)
        assert nq.dismiss(nid) is True
        assert nq.count == 0

    def test_dismiss_nonexistent(self):
        nq = NotificationQueue()
        assert nq.dismiss(999) is False

    def test_clear(self):
        nq = NotificationQueue()
        nq.add("A", duration=10.0)
        nq.add("B", duration=10.0)
        nq.clear()
        assert nq.count == 0

    def test_remaining_time(self):
        nq = NotificationQueue()
        nq.add("Timed", duration=5.0)
        result = nq.tick(2.0)
        assert result[0]["remaining"] == 3.0

    def test_priority_enum(self):
        nq = NotificationQueue()
        nq.add("Test", priority=NotificationPriority.HIGH, duration=5.0)
        result = nq.tick(0.0)
        assert result[0]["priority"] == "high"


# ===========================================================================
# KillFeed
# ===========================================================================


class TestKillFeed:
    """Tests for KillFeed."""

    def test_add_and_tick(self):
        kf = KillFeed()
        kf.add("Alpha", "Bravo", weapon="rifle", killer_alliance="friendly", victim_alliance="hostile")
        result = kf.tick(0.0)
        assert len(result) == 1
        assert result[0]["killer"] == "Alpha"
        assert result[0]["victim"] == "Bravo"
        assert result[0]["weapon"] == "rifle"

    def test_expiry(self):
        kf = KillFeed(entry_duration=3.0)
        kf.add("A", "B")
        kf.tick(2.0)
        result = kf.tick(2.0)  # age = 4.0 > 3.0
        assert len(result) == 0

    def test_max_entries(self):
        kf = KillFeed(max_entries=3, entry_duration=100.0)
        for i in range(10):
            kf.add(f"K{i}", f"V{i}")
        result = kf.tick(0.0)
        assert len(result) <= 3

    def test_colors(self):
        kf = KillFeed()
        kf.add("A", "B", killer_alliance="friendly", victim_alliance="hostile")
        result = kf.tick(0.0)
        assert result[0]["killer_color"] == "#05ffa1"
        assert result[0]["victim_color"] == "#ff2a6d"


# ===========================================================================
# HUDEngine
# ===========================================================================


class TestHUDEngine:
    """Tests for the top-level HUDEngine compositor."""

    def _world_state(self) -> dict:
        return {
            "units": [
                _unit("u1", (100, 100), "friendly"),
                _unit("u2", (400, 400), "hostile"),
            ],
            "vehicles": [_vehicle("v1")],
            "structures": [_structure("b1")],
            "camera_pos": (250.0, 250.0),
            "camera_fov": 60.0,
            "player_heading": 45.0,
            "threats": [{"position": (400, 400), "player_pos": (100, 100)}],
            "objectives": [{"position": (300, 300), "player_pos": (100, 100), "label": "OBJ-A"}],
            "economy": {"credits": 500.0, "fuel": 30.0},
            "capacity": {"credits": 1000.0, "fuel": 100.0},
            "income": {"credits": 5.0},
            "score": {"friendly": 120, "hostile": 80},
        }

    def test_render_frame_structure(self):
        hud = HUDEngine(500, 500)
        frame = hud.render_frame(self._world_state())
        assert "minimap" in frame
        assert "compass" in frame
        assert "roster" in frame
        assert "resources" in frame
        assert "notifications" in frame
        assert "kill_feed" in frame
        assert "score" in frame
        assert "frame" in frame

    def test_frame_counter(self):
        hud = HUDEngine(500, 500)
        f1 = hud.render_frame(self._world_state())
        f2 = hud.render_frame(self._world_state())
        assert f1["frame"] == 1
        assert f2["frame"] == 2

    def test_minimap_has_markers(self):
        hud = HUDEngine(500, 500)
        frame = hud.render_frame(self._world_state())
        assert len(frame["minimap"]["markers"]) == 4  # 2 units + 1 vehicle + 1 structure

    def test_compass_heading(self):
        hud = HUDEngine(500, 500)
        frame = hud.render_frame(self._world_state())
        assert frame["compass"]["heading"] == 45.0
        assert frame["compass"]["cardinal"] == "NE"

    def test_roster_filters(self):
        hud = HUDEngine(500, 500)
        frame = hud.render_frame(self._world_state(), player_alliance="friendly")
        # Only friendly units in roster
        assert frame["roster"]["count"] == 1
        assert frame["roster"]["entries"][0]["unit_id"] == "u1"

    def test_resources_rendered(self):
        hud = HUDEngine(500, 500)
        frame = hud.render_frame(self._world_state())
        assert frame["resources"]["bar_count"] == 2

    def test_score_passthrough(self):
        hud = HUDEngine(500, 500)
        frame = hud.render_frame(self._world_state())
        assert frame["score"]["friendly"] == 120

    def test_notifications_in_frame(self):
        hud = HUDEngine(500, 500)
        hud.add_notification("Contact spotted!", priority="high", duration=10.0)
        frame = hud.render_frame(self._world_state(), dt=0.016)
        assert len(frame["notifications"]) == 1
        assert frame["notifications"][0]["text"] == "Contact spotted!"

    def test_kill_feed_in_frame(self):
        hud = HUDEngine(500, 500)
        hud.add_kill("Sniper-1", "Target-3", weapon="AWP", killer_alliance="friendly")
        frame = hud.render_frame(self._world_state(), dt=0.016)
        assert len(frame["kill_feed"]) == 1

    def test_empty_world_state(self):
        hud = HUDEngine(500, 500)
        frame = hud.render_frame({})
        assert frame["minimap"]["markers"] == []
        assert frame["roster"]["count"] == 0
        assert frame["resources"]["bar_count"] == 0

    def test_add_notification_convenience(self):
        hud = HUDEngine(500, 500)
        nid = hud.add_notification("Test")
        assert isinstance(nid, int)

    def test_add_kill_convenience(self):
        hud = HUDEngine(500, 500)
        eid = hud.add_kill("A", "B")
        assert isinstance(eid, int)

    def test_multiple_frames_expire_notifications(self):
        hud = HUDEngine(500, 500)
        hud.add_notification("Flash", duration=1.0)
        f1 = hud.render_frame({}, dt=0.5)
        assert len(f1["notifications"]) == 1
        f2 = hud.render_frame({}, dt=0.6)
        assert len(f2["notifications"]) == 0  # expired at age 1.1 > 1.0

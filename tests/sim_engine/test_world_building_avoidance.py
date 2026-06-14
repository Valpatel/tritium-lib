# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""The World combat loop (`_move_unit_toward`) must not shove units through
buildings when obstacles are set.

`World._tick_units` is the lib's standalone/demo combat loop (game_server,
city_sim_backend). Unlike the live SC engine — which routes every move
through a swept collision check — this loop wrote `unit.position` directly
with no building awareness, so a retreating/flanking/seeking unit could
walk straight through a footprint.

This proves the fix: with obstacles wired via `World.set_obstacles`, the
central move helper rejects any step that would enter or cross a building
(axis-slide if one axis is clear, else hold). A unit already inside is
still allowed to move (so it can leave). With no obstacles, movement is
unchanged (regression guard).
"""

from __future__ import annotations

from tritium_lib.sim_engine.world._world import World, WorldConfig


# A minimal obstacles object in the open-SDK shape (point_in_building +
# path_crosses_building). One axis-aligned 20x20 building centered at origin.
class _Box:
    def __init__(self, cx=0.0, cy=0.0, hw=10.0, hh=10.0):
        self.cx, self.cy, self.hw, self.hh = cx, cy, hw, hh

    def point_in_building(self, x, y):
        return (self.cx - self.hw <= x <= self.cx + self.hw
                and self.cy - self.hh <= y <= self.cy + self.hh)

    def path_crosses_building(self, path):
        # Sample the polyline; true if any sample lands inside.
        for i in range(len(path) - 1):
            ax, ay = path[i]
            bx, by = path[i + 1]
            for k in range(11):
                t = k / 10.0
                if self.point_in_building(ax + (bx - ax) * t, ay + (by - ay) * t):
                    return True
        return False


def _unit_at(world, pos):
    return world.spawn_unit("infantry", "t", "friendly", pos)


def test_move_blocked_from_entering_building():
    w = World(WorldConfig())
    w.set_obstacles(_Box())
    u = _unit_at(w, (-30.0, 0.0))        # outside, left of the box
    # Target is the far side — a straight march pierces the building.
    for _ in range(200):
        before = tuple(u.position)
        w._move_unit_toward(u, (30.0, 0.0), dt=0.2)
        # It may stall against the wall; that's fine — it must never be inside.
        assert not w._obstacles.point_in_building(*u.position), (
            f"unit entered building at {u.position}")
        if tuple(u.position) == before:
            break


def test_swept_no_tunnel_through_thin_wall():
    w = World(WorldConfig())
    w.set_obstacles(_Box(hw=2.0, hh=50.0))   # thin (4m) tall wall
    u = _unit_at(w, (-20.0, 0.0))
    # Big dt * speed would tunnel a point-sample check; swept must catch it.
    crossed_to_right = False
    for _ in range(100):
        w._move_unit_toward(u, (20.0, 0.0), dt=1.0)
        assert not w._obstacles.point_in_building(*u.position)
        if u.position[0] > 2.0:
            crossed_to_right = True
            break
    assert not crossed_to_right, "unit tunneled through the thin wall"


def test_axis_slide_when_one_axis_clear():
    w = World(WorldConfig())
    w.set_obstacles(_Box())               # box edges at ±10
    u = _unit_at(w, (-30.0, 30.0))        # NW, target SE: diagonal grazes box
    moved = False
    for _ in range(300):
        before = tuple(u.position)
        w._move_unit_toward(u, (30.0, -30.0), dt=0.2)
        assert not w._obstacles.point_in_building(*u.position)
        if tuple(u.position) != before:
            moved = True
    assert moved, "axis-slide should let the unit keep making progress"


def test_unit_inside_can_leave():
    w = World(WorldConfig())
    w.set_obstacles(_Box())
    u = _unit_at(w, (0.0, 0.0))           # spawned inside (edge case)
    w._move_unit_toward(u, (100.0, 0.0), dt=0.2)
    # An inside unit is allowed to move (toward exit), so position changed.
    assert u.position != (0.0, 0.0), "unit inside a building must be able to move out"


def test_no_obstacles_is_unchanged():
    w = World(WorldConfig())               # no set_obstacles -> _obstacles None
    u = _unit_at(w, (-30.0, 0.0))
    w._move_unit_toward(u, (30.0, 0.0), dt=0.2)
    assert u.position[0] > -30.0, "without obstacles the unit must move freely"

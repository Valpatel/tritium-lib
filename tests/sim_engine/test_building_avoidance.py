# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Provable building-avoidance: units must not walk/drive through buildings.

This is the empirical regression the operator asked for: assign units to
move from A to B, over and over, across a variety of layouts and unit
types, and prove they never traverse a building footprint (unless they
are flying drones, the one intentional exception).

The load-bearing assertion is on the ACTUAL TRAVERSED POLYLINE — every
tick-to-tick segment of where the unit really went — not on the planned
waypoints. That distinction is the whole point:

  * planned waypoints can be building-free yet the unit still clips a
    building if movement cuts the corner, the turn-smoothing bulges the
    arc outside the segment, or a fast step TUNNELS through a thin wall
    between two ticks (the per-tick collision check samples only the
    destination point, so a large step can skip over a wall).

So we sweep-test the real path: for every consecutive pair of recorded
positions, assert the segment does not cross any building polygon.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.tracking.obstacles import BuildingObstacles, _segments_intersect
from tritium_lib.geo import point_in_polygon
from tritium_lib.sim_engine.core.entity import SimulationTarget
from tritium_lib.sim_engine.world.pathfinding import plan_path
from tritium_lib.sim_engine.world.terrain_map import TerrainMap


# --------------------------------------------------------------------------
# Geometry helpers (the swept test the per-tick point check does NOT do)
# --------------------------------------------------------------------------
def rect(cx: float, cy: float, w: float, h: float) -> list[tuple[float, float]]:
    """Axis-aligned building footprint centered at (cx, cy)."""
    hw, hh = w / 2.0, h / 2.0
    return [(cx - hw, cy - hh), (cx + hw, cy - hh),
            (cx + hw, cy + hh), (cx - hw, cy + hh)]


def make_obstacles(rects: list[list[tuple[float, float]]]) -> BuildingObstacles:
    obs = BuildingObstacles()
    obs.load_from_overture([{"polygon": r, "height": 8.0} for r in rects])
    return obs


def segment_hits_building(obs: BuildingObstacles,
                          ax: float, ay: float, bx: float, by: float) -> bool:
    """True if the segment A->B touches any building: an endpoint inside, the
    midpoint inside, or the segment crossing a polygon edge. This is the
    SWEPT test — it catches tunneling that a destination-point check misses."""
    if obs.point_in_building(ax, ay) or obs.point_in_building(bx, by):
        return True
    if obs.point_in_building((ax + bx) / 2.0, (ay + by) / 2.0):
        return True
    for poly in obs.polygons:
        n = len(poly)
        for j in range(n):
            cx, cy = poly[j]
            dx, dy = poly[(j + 1) % n]
            if _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return True
    return False


def first_clip(obs: BuildingObstacles,
               pts: list[tuple[float, float]]) -> tuple[int, tuple] | None:
    """Index + segment of the first traversed segment that hits a building."""
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        if segment_hits_building(obs, ax, ay, bx, by):
            return i, (pts[i], pts[i + 1])
    return None


def terrain_with_buildings(bounds: float,
                           rects: list[list[tuple[float, float]]],
                           resolution: float = 5.0) -> TerrainMap:
    """A TerrainMap with every cell covered by a building marked impassable,
    so grid A* in plan_path routes around them."""
    tm = TerrainMap(map_bounds=bounds, resolution=resolution)
    obs = make_obstacles(rects)
    step = resolution / 2.0
    n = int(2 * bounds / step) + 1
    for i in range(n):
        for j in range(n):
            x = -bounds + i * step
            y = -bounds + j * step
            if obs.point_in_building(x, y):
                tm.set_cell(x, y, "building")
    return tm


def simulate(target: SimulationTarget, dt: float = 0.1,
             max_ticks: int = 4000) -> list[tuple[float, float]]:
    """Tick the target to completion, recording its position each tick.

    Keeps the battery topped up (we are testing movement, not power) and
    stops on a terminal status, on arrival, or on a stall (no net motion
    for a while — a unit parked at a wall waiting to be re-pathed)."""
    pts: list[tuple[float, float]] = [tuple(target.position)]
    still = 0
    for _ in range(max_ticks):
        target.battery = 1.0
        prev = tuple(target.position)
        target.tick(dt)
        cur = tuple(target.position)
        pts.append(cur)
        if target.status in ("arrived", "escaped", "despawned"):
            break
        moved = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        still = still + 1 if moved < 1e-4 else 0
        if still > 30:  # parked (e.g., all waypoints blocked) — stop
            break
    return pts


# --------------------------------------------------------------------------
# 0. Primitive sanity — the geometry the whole proof rests on
# --------------------------------------------------------------------------
def test_obstacle_primitives_are_sound():
    obs = make_obstacles([rect(0, 0, 20, 20)])  # 20x20 building at origin
    assert obs.point_in_building(0, 0)            # center inside
    assert not obs.point_in_building(50, 50)      # far outside
    # A straight segment passing through the building is detected (swept).
    assert segment_hits_building(obs, -50, 0, 50, 0)
    # A segment that goes around is not.
    assert not segment_hits_building(obs, -50, 50, 50, 50)


# --------------------------------------------------------------------------
# 1. Planning: plan_path routes around buildings (no segment crosses)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("unit_type", ["rover", "person"])
def test_plan_path_routes_around_building(unit_type):
    bounds = 200.0
    wall = rect(0, 0, 20, 120)  # tall wall blocking a straight A->B
    rects = [wall]
    obs = make_obstacles(rects)
    tm = terrain_with_buildings(bounds, rects)
    start, end = (-80.0, 0.0), (80.0, 0.0)

    path = plan_path(start, end, unit_type, street_graph=None,
                     obstacles=obs, alliance="friendly", terrain_map=tm)
    assert path is not None and len(path) >= 2, "planner returned no path"
    assert not obs.path_crosses_building(path), (
        f"{unit_type} planned path crosses a building: {path}")


# --------------------------------------------------------------------------
# 2. THE PROOF: actual traversed path never clips, across a matrix
# --------------------------------------------------------------------------
LAYOUTS = {
    "wall_between": ([rect(0, 0, 16, 100)], (-80.0, 0.0), (80.0, 0.0)),
    "block_center": ([rect(0, 0, 40, 40)], (-80.0, -80.0), (80.0, 80.0)),
    "target_behind": ([rect(40, 0, 20, 60)], (-80.0, 0.0), (70.0, 0.0)),
    "two_walls": ([rect(-20, 0, 12, 90), rect(30, 10, 12, 90)],
                  (-80.0, 0.0), (80.0, 0.0)),
}
UNITS = [("rover", "friendly"), ("person", "hostile"), ("person", "neutral")]


@pytest.mark.parametrize("unit_type,alliance", UNITS)
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_traversed_path_never_clips(unit_type, alliance, layout):
    bounds = 200.0
    rects, start, end = LAYOUTS[layout]
    obs = make_obstacles(rects)
    tm = terrain_with_buildings(bounds, rects)

    path = plan_path(start, end, unit_type, street_graph=None,
                     obstacles=obs, alliance=alliance, terrain_map=tm)
    if not path:
        path = [start, end]

    target = SimulationTarget(
        target_id=f"{unit_type}-{alliance}-{layout}", name="t",
        asset_type=unit_type, alliance=alliance,
        position=start, speed=8.0, waypoints=list(path),
        is_combatant=(unit_type != "person" or alliance != "neutral"),
    )
    target.set_collision_check(obs.point_in_building)
    target.set_segment_collision_check(obs.path_crosses_building)

    pts = simulate(target)
    clip = first_clip(obs, pts)
    assert clip is None, (
        f"{unit_type}/{alliance} clipped a building in '{layout}' at "
        f"segment {clip[0]}: {clip[1]}")


# --------------------------------------------------------------------------
# 3. SKEPTICAL PROBE: fast unit + thin wall + a deliberately bad straight
#    path. The per-tick check samples only the destination point, so a big
#    step can tunnel through a thin wall. The swept assertion catches it.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("speed", [8.0, 30.0, 80.0])
def test_enforcement_blocks_tunneling_through_thin_wall(speed):
    obs = make_obstacles([rect(0, 0, 4, 200)])  # 4m-thin, 200m-tall wall
    start, end = (-60.0, 0.0), (60.0, 0.0)       # straight line pierces it
    target = SimulationTarget(
        target_id=f"tunnel-{speed}", name="t", asset_type="rover", alliance="friendly",
        position=start, speed=speed, waypoints=[start, end],
        is_combatant=True,
    )
    target.set_collision_check(obs.point_in_building)
    target.set_segment_collision_check(obs.path_crosses_building)
    pts = simulate(target)
    clip = first_clip(obs, pts)
    assert clip is None, (
        f"rover at {speed} m/s TUNNELED through the thin wall at segment "
        f"{clip[0]}: {clip[1]} — per-tick collision check is point-sampled, "
        f"not swept")


# --------------------------------------------------------------------------
# 4. "Over and over": repeated A->B->C->D reassignment in a building field
# --------------------------------------------------------------------------
def test_repeated_reassignment_no_clip():
    bounds = 200.0
    rects = [rect(-40, -40, 30, 30), rect(40, 40, 30, 30),
             rect(-40, 40, 25, 25), rect(40, -40, 25, 25), rect(0, 0, 24, 24)]
    obs = make_obstacles(rects)
    tm = terrain_with_buildings(bounds, rects)
    waypoints_corners = [(-90.0, -90.0), (90.0, -90.0),
                         (90.0, 90.0), (-90.0, 90.0), (-90.0, -90.0)]

    target = SimulationTarget(
        target_id="patrol", name="patrol", asset_type="rover", alliance="friendly",
        position=waypoints_corners[0], speed=10.0, is_combatant=True,
    )
    target.set_collision_check(obs.point_in_building)
    target.set_segment_collision_check(obs.path_crosses_building)

    for leg in range(len(waypoints_corners) - 1):
        a, b = waypoints_corners[leg], waypoints_corners[leg + 1]
        path = plan_path(a, b, "rover", street_graph=None, obstacles=obs,
                         alliance="friendly", terrain_map=tm) or [a, b]
        target.position = a
        target.status = "active"
        target.waypoints = list(path)
        if target.movement is not None:
            target.movement.set_path(list(path))
        pts = simulate(target)
        clip = first_clip(obs, pts)
        assert clip is None, (
            f"leg {leg} ({a}->{b}) clipped at segment {clip[0]}: {clip[1]}")


# --------------------------------------------------------------------------
# 5. The intentional exception: drones fly OVER buildings (documented)
# --------------------------------------------------------------------------
def test_drone_is_an_intentional_flyover():
    obs = make_obstacles([rect(0, 0, 40, 40)])
    start, end = (-80.0, 0.0), (80.0, 0.0)
    path = plan_path(start, end, "drone", street_graph=None, obstacles=obs,
                     alliance="friendly", terrain_map=None)
    # Drones get a straight line — they are EXPECTED to overfly footprints.
    assert path == [start, end], "drones should fly straight (over buildings)"

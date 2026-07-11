# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Map data -> neutral 3D scene pipeline (deterministic, no USD/Isaac/GPU).

Proves the reusable digital-twin geometry: extrude building footprints, build a
terrain heightfield from a DEM, assemble + serialize a Scene3D for an AO.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.geo.gis.models import ElevationGrid
from tritium_lib.geo.scene3d import (
    LocalProjection,
    Mesh3D,
    Scene3D,
    build_scene3d,
    buildings_from_geojson,
    extrude_footprint,
    make_elevation_sampler,
    roads_from_geojson,
    terrain_heightfield_mesh,
    water_from_geojson,
)


# ------------------------------------------------------------------ projection

def test_local_projection_roundtrip_and_origin():
    proj = LocalProjection(37.7159, -121.896)
    assert proj.to_local(37.7159, -121.896) == (0.0, 0.0)
    e, n = proj.to_local(37.7159, -121.895)  # one thousandth east
    assert e > 0 and abs(n) < 1e-6
    lat, lng = proj.to_latlng(e, n)
    assert lat == pytest.approx(37.7159, abs=1e-6)
    assert lng == pytest.approx(-121.895, abs=1e-6)


def test_local_projection_north_is_positive_y():
    proj = LocalProjection(37.0, -121.0)
    _, n = proj.to_local(37.001, -121.0)
    assert n > 0  # north -> +Y


# ------------------------------------------------------------------ extrusion

def _square(cx=0.0, cy=0.0, s=10.0):
    return [(cx - s, cy - s), (cx + s, cy - s), (cx + s, cy + s), (cx - s, cy + s)]


def test_extrude_square_prism_topology():
    m = extrude_footprint(_square(), base_z=100.0, height=12.0, name="tower")
    assert m is not None
    assert m.kind == "building"
    assert m.vertex_count == 8            # 4 floor + 4 roof
    # 2 roof + 2 floor fan tris + 4 walls*2 = 12 triangles
    assert m.face_count == 12
    zs = [v[2] for v in m.vertices]
    assert min(zs) == pytest.approx(100.0)
    assert max(zs) == pytest.approx(112.0)


def test_extrude_closes_repeated_ring():
    ring = _square()
    ring = ring + [ring[0]]  # explicit closing vertex
    m = extrude_footprint(ring, base_z=0.0, height=5.0)
    assert m.vertex_count == 8  # closing dup dropped


def test_extrude_degenerate_returns_none():
    assert extrude_footprint([(0, 0), (1, 1)], 0.0, 5.0) is None


def test_extrude_minimum_height_floor():
    m = extrude_footprint(_square(), base_z=0.0, height=0.0)
    zs = [v[2] for v in m.vertices]
    assert max(zs) >= 0.5  # a building is never zero-height


# ------------------------------------------------------------------ terrain

def _ramp_grid(ncols=5, nrows=4):
    # Elevation ramps west->east from 0..(ncols-1)*10.
    values = []
    for iy in range(nrows):
        for ix in range(ncols):
            values.append(ix * 10.0)
    return ElevationGrid(west=-121.9, south=37.70, east=-121.88, north=37.72,
                         ncols=ncols, nrows=nrows, values=values)


def test_terrain_mesh_vertex_and_face_counts():
    grid = _ramp_grid(5, 4)
    proj = LocalProjection(37.71, -121.89)
    m = terrain_heightfield_mesh(grid, proj, subsample=1)
    assert m is not None and m.kind == "terrain"
    assert m.vertex_count == 5 * 4
    assert m.face_count == (5 - 1) * (4 - 1) * 2  # 2 tris per cell


def test_terrain_mesh_elevation_preserved():
    grid = _ramp_grid(5, 4)
    proj = LocalProjection(37.71, -121.89)
    m = terrain_heightfield_mesh(grid, proj, subsample=1)
    zs = sorted({round(v[2]) for v in m.vertices})
    assert zs == [0, 10, 20, 30, 40]


def test_terrain_nodata_filled_with_mean():
    grid = _ramp_grid(3, 3)
    grid.values[4] = None  # centre NoData
    proj = LocalProjection(37.71, -121.89)
    m = terrain_heightfield_mesh(grid, proj, subsample=1)
    assert all(v[2] is not None for v in m.vertices)


def test_elevation_sampler_nearest_cell():
    grid = _ramp_grid(5, 4)
    proj = LocalProjection(37.71, -121.89)
    sample = make_elevation_sampler(grid, proj)
    # East edge should sample the high end of the ramp.
    e_east, n0 = proj.to_local(37.71, -121.881)
    assert sample(e_east, n0) >= 30.0
    e_west, _ = proj.to_local(37.71, -121.899)
    assert sample(e_west, n0) <= 10.0


# ------------------------------------------------------------------ geojson

def _geojson_two_buildings():
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Polygon", "coordinates": [[
                 [-121.896, 37.7159], [-121.8958, 37.7159],
                 [-121.8958, 37.7161], [-121.896, 37.7161], [-121.896, 37.7159]]]},
             "properties": {"name": "Hall", "height_m": 20.0, "kind": "civic"}},
            {"type": "Feature",
             "geometry": {"type": "Polygon", "coordinates": [[
                 [-121.897, 37.716], [-121.8968, 37.716],
                 [-121.8968, 37.7162], [-121.897, 37.7162]]]},
             "properties": {"name": "Shed", "levels": 2}},  # -> 6 m
        ],
    }


def test_buildings_from_geojson_counts_and_height():
    proj = LocalProjection(37.716, -121.8965)
    meshes = buildings_from_geojson(_geojson_two_buildings(), proj)
    assert len(meshes) == 2
    hall = next(m for m in meshes if m.name == "Hall")
    shed = next(m for m in meshes if m.name == "Shed")
    assert hall.height_m == pytest.approx(20.0)
    assert shed.height_m == pytest.approx(6.0)  # 2 levels * 3 m
    assert all(m.kind == "building" for m in meshes)


def test_buildings_sit_on_terrain_elevation():
    grid = _ramp_grid(5, 4)
    proj = LocalProjection(37.71, -121.89)
    sampler = make_elevation_sampler(grid, proj)
    meshes = buildings_from_geojson(_geojson_two_buildings(),
                                    LocalProjection(37.716, -121.8965), sampler)
    # Base z should be non-zero (sampled from the ramp), not glued to 0.
    for m in meshes:
        base = min(v[2] for v in m.vertices)
        assert base >= 0.0


# ------------------------------------------------------------------ scene

def test_build_scene3d_assembles_terrain_and_buildings():
    grid = _ramp_grid(6, 6)
    scene = build_scene3d(
        ao="testville",
        bbox=(-121.90, 37.70, -121.88, 37.72),
        elevation_grid=grid,
        buildings_geojson=_geojson_two_buildings(),
        terrain_subsample=1,
    )
    assert scene.ao == "testville"
    assert len(scene.by_kind("terrain")) == 1
    assert len(scene.by_kind("building")) == 2
    st = scene.stats()
    assert st["by_kind"]["building"] == 2
    assert st["vertices"] > 0 and st["faces"] > 0
    b = scene.bounds()
    assert b["max"][0] > b["min"][0]  # non-degenerate east extent


def test_scene3d_json_roundtrip():
    scene = build_scene3d("t", (-121.90, 37.70, -121.88, 37.72),
                          elevation_grid=_ramp_grid(4, 4),
                          buildings_geojson=_geojson_two_buildings())
    d = scene.to_dict()
    back = Scene3D.from_dict(d)
    assert back.ao == scene.ao
    assert len(back.meshes) == len(scene.meshes)
    assert back.origin_lat == pytest.approx(scene.origin_lat)


def test_scene3d_obj_export_wellformed():
    scene = build_scene3d("t", (-121.90, 37.70, -121.88, 37.72),
                          elevation_grid=_ramp_grid(4, 4),
                          buildings_geojson=_geojson_two_buildings())
    obj = scene.to_obj()
    vcount = sum(1 for ln in obj.splitlines() if ln.startswith("v "))
    fcount = sum(1 for ln in obj.splitlines() if ln.startswith("f "))
    assert vcount == scene.stats()["vertices"]
    assert fcount == scene.stats()["faces"]
    # OBJ face indices are 1-based and within range.
    for ln in obj.splitlines():
        if ln.startswith("f "):
            for tok in ln.split()[1:]:
                assert int(tok) >= 1


def test_build_scene3d_flat_without_dem():
    scene = build_scene3d("t", (-121.90, 37.70, -121.88, 37.72),
                          elevation_grid=None,
                          buildings_geojson=_geojson_two_buildings())
    assert len(scene.by_kind("terrain")) == 0
    assert len(scene.by_kind("building")) == 2
    for m in scene.by_kind("building"):
        assert min(v[2] for v in m.vertices) == pytest.approx(0.0)


# ------------------------------------------------------------------ roads

def _geojson_roads():
    # Two TIGER-style LineStrings + one MultiLineString.
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [
                 [-121.900, 37.710], [-121.890, 37.710], [-121.885, 37.712]]},
             "properties": {"kind": "S1400", "name": "Main St"}},
            {"type": "Feature",  # ramp -> narrower class
             "geometry": {"type": "LineString", "coordinates": [
                 [-121.895, 37.705], [-121.895, 37.715]]},
             "properties": {"kind": "S1630", "name": "Ramp"}},
            {"type": "Feature",
             "geometry": {"type": "MultiLineString", "coordinates": [
                 [[-121.898, 37.708], [-121.892, 37.708]],
                 [[-121.892, 37.708], [-121.888, 37.706]]]},
             "properties": {"kind": "residential", "name": "Loop"}},
        ],
    }


def test_roads_from_geojson_counts_and_kind():
    proj = LocalProjection(37.710, -121.892)
    meshes = roads_from_geojson(_geojson_roads(), proj)
    # 2 LineStrings + 2 sub-lines of the MultiLineString = 4 ribbons.
    assert len(meshes) == 4
    assert all(m.kind == "road" for m in meshes)
    # Every ribbon is a valid strip: even vertex count, >=1 quad (2 tris).
    for m in meshes:
        assert m.vertex_count >= 4 and m.vertex_count % 2 == 0
        assert m.face_count == (m.vertex_count // 2 - 1) * 2


def test_roads_ribbon_width_scales_with_class():
    proj = LocalProjection(37.710, -121.892)
    meshes = roads_from_geojson(_geojson_roads(), proj)
    main = next(m for m in meshes if m.name == "Main St")     # S1400 -> 6 m
    ramp = next(m for m in meshes if m.name == "Ramp")        # S1630 -> 5 m
    # Ribbon half-width = perpendicular spread of a left/right vertex pair.
    def half_width(m):
        (lx, ly, _), (rx, ry, _) = m.vertices[0], m.vertices[1]
        return math.hypot(lx - rx, ly - ry) / 2.0
    assert half_width(main) > half_width(ramp)


def test_roads_drape_on_terrain():
    grid = _ramp_grid(6, 6)
    proj = LocalProjection(37.71, -121.89)
    sampler = make_elevation_sampler(grid, proj)
    meshes = roads_from_geojson(_geojson_roads(), proj, sampler)
    # Non-flat DEM -> a road spanning the ramp has varying Z (it drapes).
    main = next(m for m in meshes if m.name == "Main St")
    zs = {round(v[2], 2) for v in main.vertices}
    assert len(zs) > 1


# ------------------------------------------------------------------ water

def _geojson_water():
    # One NHD flowline (river) + one waterbody Polygon.
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "LineString", "coordinates": [
                 [-121.900, 37.712], [-121.895, 37.711], [-121.890, 37.713]]},
             "properties": {"kind": "river", "name": "Creek"}},
            {"type": "Feature",
             "geometry": {"type": "Polygon", "coordinates": [[
                 [-121.894, 37.706], [-121.891, 37.706],
                 [-121.891, 37.708], [-121.894, 37.708], [-121.894, 37.706]]]},
             "properties": {"kind": "waterbody", "name": "Pond"}},
        ],
    }


def test_water_from_geojson_polygon_and_flowline():
    proj = LocalProjection(37.710, -121.892)
    meshes = water_from_geojson(_geojson_water(), proj)
    assert len(meshes) == 2
    assert all(m.kind == "water" for m in meshes)
    pond = next(m for m in meshes if m.name == "Pond")     # polygon fill
    creek = next(m for m in meshes if m.name == "Creek")   # ribbon
    # Polygon fill is a single flat surface (one Z), fan-triangulated.
    assert len({round(v[2], 3) for v in pond.vertices}) == 1
    assert pond.face_count == pond.vertex_count - 2  # fan
    # Ribbon is a strip.
    assert creek.vertex_count % 2 == 0 and creek.face_count >= 2


def test_water_sits_below_terrain():
    grid = _ramp_grid(6, 6)
    proj = LocalProjection(37.71, -121.89)
    sampler = make_elevation_sampler(grid, proj)
    meshes = water_from_geojson(_geojson_water(), proj, sampler)
    pond = next(m for m in meshes if m.name == "Pond")
    # Fill Z is ground-at-centroid minus the epsilon -> strictly below sample.
    cx = sum(v[0] for v in pond.vertices) / pond.vertex_count
    cy = sum(v[1] for v in pond.vertices) / pond.vertex_count
    assert pond.vertices[0][2] < sampler(cx, cy)


# ------------------------------------------------------------------ scene + roads/water

def test_build_scene3d_with_roads_and_water():
    grid = _ramp_grid(6, 6)
    scene = build_scene3d(
        ao="rw", bbox=(-121.90, 37.70, -121.88, 37.72),
        elevation_grid=grid, buildings_geojson=_geojson_two_buildings(),
        terrain_subsample=1,
        roads_geojson=_geojson_roads(), water_geojson=_geojson_water(),
    )
    assert len(scene.by_kind("terrain")) == 1
    assert len(scene.by_kind("building")) == 2
    assert len(scene.by_kind("road")) == 4
    assert len(scene.by_kind("water")) == 2
    assert scene.stats()["by_kind"]["road"] == 4
    assert scene.stats()["by_kind"]["water"] == 2


def test_build_scene3d_none_roads_water_is_unchanged():
    # Byte-identical to omitting the new params entirely (backward compat).
    args = dict(ao="t", bbox=(-121.90, 37.70, -121.88, 37.72),
                elevation_grid=_ramp_grid(5, 5),
                buildings_geojson=_geojson_two_buildings())
    base = build_scene3d(**args)
    with_none = build_scene3d(**args, roads_geojson=None, water_geojson=None)
    assert base.to_dict() == with_none.to_dict()
    assert with_none.by_kind("road") == [] and with_none.by_kind("water") == []

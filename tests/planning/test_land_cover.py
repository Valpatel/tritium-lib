# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for real-terrain mobility stamping in tritium_lib.planning.

Covers the costmap lane's NLCD land-cover + NHD hydrography integration:

    - :meth:`CostmapBuilder.add_land_cover` — forest/wetland become soft-cost
      zones (slower), OPEN WATER becomes lethal, open/developed stay normal,
      and the ``mobility_fn`` SEASONAL SEAM modulates the per-category cost.
    - :meth:`CostmapBuilder.add_water_obstacles` — NHD waterbodies + wide
      rivers stamp lethal, narrow streams/canals stay traversable, and a road
      over water stays passable (the BRIDGE seam).
    - the real USGS NHD Boulder fixture drives a lethal-water stamp.

Deterministic: a fake linear projector stands in for the geo singleton so the
expected per-cell cost stamps are exact.
"""

import json
from pathlib import Path

import pytest

from tritium_lib.planning.costmap import (
    NHD_RIVER_HALF_WIDTH_M,
    CostmapBuilder,
)
from tritium_lib.geo.gis.landcover import LandCoverGrid, tactical_profile


# Identity projector: lon -> x, lat -> y (the grid bbox is already in "meters").
def _identity(lon, lat):
    return (float(lon), float(lat))


# A 400 m x 400 m world at 5 m resolution: 80 x 80 cells.
BOUNDS = (0.0, 0.0, 400.0, 400.0)


def _grid(codes, ncols, nrows, bounds=BOUNDS):
    """Build a LandCoverGrid tiling ``bounds`` with the given class codes."""
    w, s, e, n = bounds
    return LandCoverGrid(west=w, south=s, east=e, north=n,
                         ncols=ncols, nrows=nrows, codes=list(codes))


# ---------------------------------------------------------------------------
# add_land_cover — mobility zones + lethal water
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLandCoverMobility:
    def test_forest_becomes_soft_cost_zone(self):
        # Whole grid evergreen forest (code 42, mobility 3.0).
        grid = _grid([42] * 16, 4, 4)
        assert tactical_profile(42)["mobility_cost"] == 3.0
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        summary = b.add_land_cover(grid, to_local=_identity)
        cm = b.build()
        # A cell in the interior: base_cost(1.0) * mobility(3.0) = 3.0.
        col, row = cm.world_to_grid(200.0, 200.0)
        assert cm.cost_at(col, row) == pytest.approx(3.0)
        assert summary["slow"] > 0
        assert summary["water"] == 0

    def test_open_water_is_lethal(self):
        grid = _grid([11] * 16, 4, 4)  # all Open Water
        assert tactical_profile(11)["passable"] is False
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        summary = b.add_land_cover(grid, to_local=_identity)
        cm = b.build()
        col, row = cm.world_to_grid(200.0, 200.0)
        assert cm.is_lethal(col, row)
        assert summary["water"] > 0
        assert summary["slow"] == 0

    def test_open_and_developed_stay_normal(self):
        # Developed Open Space (21, mobility 1.0) + Grassland (71, mobility 1.1).
        grid = _grid([21, 21, 71, 71] * 4, 4, 4)
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        b.add_land_cover(grid, to_local=_identity)
        cm = b.build()
        # Column 0 (x~50) is developed-open -> no zone -> base 1.0.
        col, row = cm.world_to_grid(50.0, 200.0)
        assert cm.cost_at(col, row) == pytest.approx(1.0)

    def test_forest_patch_detours_route(self):
        # A vertical forest wall down the middle third; a route from west to
        # east must be MORE expensive through it than the open-cell base sum.
        ncols = nrows = 8
        codes = []
        for _iy in range(nrows):
            for ix in range(ncols):
                codes.append(42 if 3 <= ix <= 4 else 21)  # forest band cols 3-4
        grid = _grid(codes, ncols, nrows)
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        b.add_land_cover(grid, to_local=_identity)
        cm = b.build()
        # A cell squarely in the forest band (x ~ 175-225) is 3x an open cell.
        fcol, frow = cm.world_to_grid(200.0, 200.0)
        ocol, orow = cm.world_to_grid(50.0, 200.0)
        assert cm.cost_at(fcol, frow) == pytest.approx(3.0)
        assert cm.cost_at(ocol, orow) == pytest.approx(1.0)

    def test_water_over_road_is_a_bridge(self):
        # A road first, then a water cell covering the road stays passable.
        grid = _grid([11] * 16, 4, 4)
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        road_fc = {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[0, 200], [400, 200]]},
            "properties": {"source": "tiger", "kind": "S1100"}}]}
        b.add_gis_features(road_fc, to_local=_identity)
        summary = b.add_land_cover(grid, to_local=_identity)
        cm = b.build()
        # On the road line (y=200) -> passable bridge; off it -> lethal water.
        bcol, brow = cm.world_to_grid(200.0, 200.0)
        wcol, wrow = cm.world_to_grid(200.0, 40.0)
        assert not cm.is_lethal(bcol, brow), "road over water must be a bridge"
        assert cm.is_lethal(wcol, wrow)
        assert summary["bridges"] > 0

    def test_nodata_cells_are_neutral(self):
        # None codes -> neutral profile (mobility 1.0, passable) -> no stamp.
        grid = _grid([None] * 16, 4, 4)
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        summary = b.add_land_cover(grid, to_local=_identity)
        cm = b.build()
        col, row = cm.world_to_grid(200.0, 200.0)
        assert cm.cost_at(col, row) == pytest.approx(1.0)
        assert summary == {"water": 0, "slow": 0, "bridges": 0, "cells": 0}

    def test_degenerate_grid_is_noop(self):
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        assert b.add_land_cover(_grid([], 0, 0), to_local=_identity)["cells"] == 0
        # Duck-typed object missing tactical_field() -> graceful empty.
        class Bad:
            ncols = 4
            nrows = 4
        assert b.add_land_cover(Bad(), to_local=_identity)["cells"] == 0

    def test_seasonal_mobility_fn_seam(self):
        # The environment lane's hook: raise open-ground cost in snow.
        grid = _grid([21] * 16, 4, 4)  # developed-open, base mobility 1.0

        def winter(category, mobility):
            # Snow makes even open ground slow.
            return mobility * 2.5 if category == "developed" else mobility

        b = CostmapBuilder(BOUNDS, resolution=5.0)
        summary = b.add_land_cover(grid, to_local=_identity, mobility_fn=winter)
        cm = b.build()
        col, row = cm.world_to_grid(200.0, 200.0)
        assert cm.cost_at(col, row) == pytest.approx(2.5)
        assert summary["slow"] > 0

    def test_seasonal_fn_never_downgrades_water(self):
        # Even a mischievous season hook cannot un-lethal open water.
        grid = _grid([11] * 16, 4, 4)
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        b.add_land_cover(grid, to_local=_identity,
                         mobility_fn=lambda cat, m: 0.1)
        cm = b.build()
        col, row = cm.world_to_grid(200.0, 200.0)
        assert cm.is_lethal(col, row)


# ---------------------------------------------------------------------------
# add_water_obstacles — NHD waterbodies + rivers, bridges
# ---------------------------------------------------------------------------

def _nhd_fc(features):
    return {"type": "FeatureCollection", "features": features}


def _line(coords, kind, **props):
    return {"type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"source": "nhd", "kind": kind, **props}}


def _poly(ring, kind, **props):
    return {"type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring]},
            "properties": {"source": "nhd", "kind": kind, **props}}


@pytest.mark.unit
class TestNhdWater:
    def test_waterbody_is_lethal(self):
        fc = _nhd_fc([_poly([[100, 100], [300, 100], [300, 300], [100, 300],
                             [100, 100]], "waterbody")])
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        summary = b.add_water_obstacles(fc, to_local=_identity)
        cm = b.build()
        col, row = cm.world_to_grid(200.0, 200.0)
        assert cm.is_lethal(col, row)
        assert summary["waterbody"] == 1
        assert summary["cells"] > 0

    def test_river_is_lethal_corridor(self):
        fc = _nhd_fc([_line([[200, 0], [200, 400]], "river")])
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        summary = b.add_water_obstacles(fc, to_local=_identity)
        cm = b.build()
        col, row = cm.world_to_grid(200.0, 200.0)
        assert cm.is_lethal(col, row)
        assert summary["river"] == 1
        # Corridor half-width default: a cell one bank away (< half) is lethal,
        # a cell well clear of the river is not.
        near = cm.world_to_grid(200.0 + NHD_RIVER_HALF_WIDTH_M - 1.0, 200.0)
        far = cm.world_to_grid(200.0 + 40.0, 200.0)
        assert cm.is_lethal(*near)
        assert not cm.is_lethal(*far)

    def test_stream_and_canal_stay_traversable(self):
        # Default impassable_kinds is {"river"} only.
        fc = _nhd_fc([
            _line([[100, 0], [100, 400]], "stream"),
            _line([[300, 0], [300, 400]], "canal"),
        ])
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        summary = b.add_water_obstacles(fc, to_local=_identity)
        cm = b.build()
        assert not cm.is_lethal(*cm.world_to_grid(100.0, 200.0))
        assert not cm.is_lethal(*cm.world_to_grid(300.0, 200.0))
        assert summary["cells"] == 0
        assert summary["river"] == 0

    def test_stream_lethal_when_opted_in(self):
        fc = _nhd_fc([_line([[200, 0], [200, 400]], "stream")])
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        summary = b.add_water_obstacles(
            fc, to_local=_identity, impassable_kinds=frozenset({"stream"}))
        cm = b.build()
        assert cm.is_lethal(*cm.world_to_grid(200.0, 200.0))
        assert summary["river"] == 1  # counted under the flowline tally

    def test_road_over_river_is_a_bridge(self):
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        road_fc = {"type": "FeatureCollection", "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[0, 200], [400, 200]]},
            "properties": {"source": "tiger", "kind": "S1100"}}]}
        b.add_gis_features(road_fc, to_local=_identity)
        fc = _nhd_fc([_line([[200, 0], [200, 400]], "river")])
        summary = b.add_water_obstacles(fc, to_local=_identity)
        cm = b.build()
        # The road/river crossing (200, 200) stays passable (bridge).
        assert not cm.is_lethal(*cm.world_to_grid(200.0, 200.0))
        # The river away from the road is still lethal.
        assert cm.is_lethal(*cm.world_to_grid(200.0, 40.0))
        assert summary["bridges"] > 0

    def test_explicit_width_overrides_default(self):
        # A wide river (width_m 60 -> half 30) blocks farther than the default.
        fc = _nhd_fc([_line([[200, 0], [200, 400]], "river", width_m=60.0)])
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        b.add_water_obstacles(fc, to_local=_identity)
        cm = b.build()
        # 25 m off-center is within the 30 m half-width -> lethal.
        assert cm.is_lethal(*cm.world_to_grid(225.0, 200.0))


# ---------------------------------------------------------------------------
# sever_crossings — a blown bridge denies the span; the planner re-routes
# ---------------------------------------------------------------------------


def _road_ew(y):
    """An E-W TIGER road spanning the world at latitude ``y`` (a bridge where it
    crosses the N-S river)."""
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[0, y], [400, y]]},
        "properties": {"source": "tiger", "kind": "S1100"}}]}


def _crosses_wall_near(path, y_expected, tol=40.0):
    """True if the route crosses the x=200 river wall within ``tol`` m of ``y``."""
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        if (x0 - 200.0) * (x1 - 200.0) <= 0 and x0 != x1:
            t = (200.0 - x0) / (x1 - x0)
            yc = y0 + t * (y1 - y0)
            if abs(yc - y_expected) <= tol:
                return True
    return False


@pytest.mark.unit
class TestSeverCrossings:
    """A river wall split by two road bridges; severing one re-routes the plan."""

    def _two_bridge_builder(self):
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        b.add_gis_features(_road_ew(100.0), to_local=_identity)   # bridge A
        b.add_gis_features(_road_ew(300.0), to_local=_identity)   # bridge B
        b.add_water_obstacles(
            _nhd_fc([_line([[200, 0], [200, 400]], "river")]),
            to_local=_identity)
        return b

    def test_intact_route_uses_the_near_bridge(self):
        from tritium_lib.planning.astar import plan_route
        cm = self._two_bridge_builder().build()
        r = plan_route(cm, (40.0, 100.0), (360.0, 100.0), clearance_m=0.0)
        assert r.success
        # It crosses the wall at bridge A (y=100), not the far bridge B.
        assert _crosses_wall_near(r.path, 100.0)

    def test_severing_near_bridge_reroutes_to_far_bridge(self):
        from tritium_lib.planning.astar import plan_route
        b = self._two_bridge_builder()
        # Blow bridge A (the whole ~span, radius covers road+river width).
        sv = b.sever_crossings([(200.0, 100.0)], radius_m=20.0)
        assert sv["severed"] > 0
        cm = b.build()
        # Bridge A cell is now lethal (severed) ...
        assert cm.is_lethal(*cm.world_to_grid(200.0, 100.0))
        # ... but bridge B survives, so the route still solves — via y=300.
        r = plan_route(cm, (40.0, 100.0), (360.0, 100.0), clearance_m=0.0)
        assert r.success
        assert _crosses_wall_near(r.path, 300.0)
        assert not _crosses_wall_near(r.path, 100.0)

    def test_severing_the_only_bridge_denies_the_route(self):
        from tritium_lib.planning.astar import plan_route
        b = CostmapBuilder(BOUNDS, resolution=5.0)
        b.add_gis_features(_road_ew(100.0), to_local=_identity)  # sole bridge
        b.add_water_obstacles(
            _nhd_fc([_line([[200, 0], [200, 400]], "river")]),
            to_local=_identity)
        intact = plan_route(
            b.build(), (40.0, 100.0), (360.0, 100.0), clearance_m=0.0)
        assert intact.success  # the lone bridge carries it while intact

        b.sever_crossings([(200.0, 100.0)], radius_m=20.0)
        denied = plan_route(
            b.build(), (40.0, 100.0), (360.0, 100.0), clearance_m=0.0)
        assert not denied.success  # blown -> no crossing -> no path

    def test_repair_is_a_fresh_build_without_the_sever(self):
        # Sever is per-build state, not a permanent global: the engine rebuilds a
        # FRESH builder each costmap cycle, so "repair" == the next rebuild omits
        # the sever.  Two independent builders over the same geometry: one blown,
        # one intact.
        from tritium_lib.planning.astar import plan_route

        def _sole_bridge_builder():
            b = CostmapBuilder(BOUNDS, resolution=5.0)
            b.add_gis_features(_road_ew(100.0), to_local=_identity)
            b.add_water_obstacles(
                _nhd_fc([_line([[200, 0], [200, 400]], "river")]),
                to_local=_identity)
            return b

        blown = _sole_bridge_builder()
        blown.sever_crossings([(200.0, 100.0)], radius_m=20.0)
        assert not plan_route(
            blown.build(), (40.0, 100.0), (360.0, 100.0), clearance_m=0.0).success

        repaired = _sole_bridge_builder()  # fresh build, no sever applied
        assert plan_route(
            repaired.build(), (40.0, 100.0), (360.0, 100.0), clearance_m=0.0).success


# ---------------------------------------------------------------------------
# Real USGS NHD fixture drives a lethal-water stamp
# ---------------------------------------------------------------------------

_NHD_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "src/tritium_lib/geo/gis/fixtures/nhd_hydro_boulder.json"
)


@pytest.mark.unit
class TestRealNhdFixture:
    def test_boulder_fixture_stamps_water(self):
        assert _NHD_FIXTURE.exists(), _NHD_FIXTURE
        fc = json.loads(_NHD_FIXTURE.read_text())
        feats = fc.get("features", [])
        assert feats, "fixture must carry features"
        # Fixture bbox (lon/lat); use a linear projector so cells resolve.
        lons = []
        lats = []
        for f in feats:
            _walk_coords(f["geometry"]["coordinates"], lons, lats)
        w, e = min(lons), max(lons)
        s, n = min(lats), max(lats)
        # ~111 km / deg; small AO -> scale to a few-km local frame.
        scale = 111_000.0

        def proj(lon, lat):
            return ((lon - w) * scale, (lat - s) * scale)

        bounds = (0.0, 0.0, (e - w) * scale, (n - s) * scale)
        b = CostmapBuilder(bounds, resolution=20.0)
        summary = b.add_water_obstacles(fc, to_local=proj)
        # The Boulder fixture carries waterbodies AND named rivers.
        assert summary["waterbody"] >= 1
        assert summary["river"] >= 1
        assert summary["cells"] > 0
        cm = b.build()
        lethal = sum(1 for r in cm.grid for c in r if c == cm.LETHAL)
        assert lethal > 0, "real NHD water must produce lethal cells"


def _walk_coords(coords, lons, lats):
    """Recursively collect lon/lat from a GeoJSON coordinate array."""
    if (isinstance(coords, (list, tuple)) and len(coords) == 2
            and all(isinstance(v, (int, float)) for v in coords)):
        lons.append(float(coords[0]))
        lats.append(float(coords[1]))
        return
    for c in coords:
        _walk_coords(c, lons, lats)

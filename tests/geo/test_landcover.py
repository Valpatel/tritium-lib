# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.gis.landcover + NlcdLandCoverFetcher.

Covers the canonical NLCD class table, nearest-colour classification, the
per-class tactical doctrine, the LandCoverGrid raster (geometry identical to
ElevationGrid, round-trip, category rollups), and the fetcher's degradation
chain (fixture-intersect + empty).  No network.

The PNG-decode path (``parse_png_grid``) needs Pillow, which is NOT a lib
dependency — those tests ``importorskip("PIL")`` so the suite is green with or
without it.  Everything else is pure stdlib.
"""

import io

import pytest

from tritium_lib.geo.gis import (
    NLCD_CLASSES,
    GeoBBox,
    LandCoverClass,
    LandCoverGrid,
    NlcdLandCoverFetcher,
    classify_rgb,
    tactical_profile,
)
from tritium_lib.geo.gis import fetchers as fetchers_mod
from tritium_lib.geo.gis.models import ElevationGrid

# The two packaged AO bboxes.
DUBLIN = GeoBBox(-121.912, 37.704, -121.880, 37.728)
BOULDER = GeoBBox(-105.30, 39.98, -105.26, 40.02)


# ---------------------------------------------------------------------------
# Canonical class table
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNlcdClasses:
    def test_all_expected_codes_present(self):
        expected = {11, 12, 21, 22, 23, 24, 31, 41, 42, 43,
                    51, 52, 71, 72, 73, 74, 81, 82, 90, 95}
        assert set(NLCD_CLASSES) == expected

    def test_every_class_is_well_formed(self):
        for code, cls in NLCD_CLASSES.items():
            assert isinstance(cls, LandCoverClass)
            assert cls.code == code
            assert cls.name
            assert isinstance(cls.rgb, tuple) and len(cls.rgb) == 3
            assert all(0 <= c <= 255 for c in cls.rgb)
            assert cls.category
            assert 0.0 <= cls.cover <= 1.0
            assert 0.0 <= cls.concealment <= 1.0
            assert cls.mobility_cost >= 1.0
            assert isinstance(cls.passable, bool)

    def test_canonical_rgb_spot_checks(self):
        assert NLCD_CLASSES[11].rgb == (70, 107, 159)
        assert NLCD_CLASSES[42].rgb == (28, 95, 44)
        assert NLCD_CLASSES[23].rgb == (235, 0, 0)


# ---------------------------------------------------------------------------
# classify_rgb — exact + nearest-colour
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestClassifyRgb:
    def test_exact_canonical_colours_map_to_own_code(self):
        for code, cls in NLCD_CLASSES.items():
            assert classify_rgb(*cls.rgb) == code

    def test_near_colour_developed_medium(self):
        # The MRLC WMS renders Developed Medium as (237,0,0), not the canonical
        # (235,0,0): nearest-colour must still land on 23.
        assert classify_rgb(237, 0, 0) == 23

    def test_near_colour_evergreen(self):
        # A slightly-off evergreen green stays on 42 (28,95,44), not 41 (104,171,95).
        assert classify_rgb(26, 93, 42) == 42

    def test_near_colour_open_water(self):
        assert classify_rgb(68, 105, 160) == 11

    def test_pure_black_and_white_do_not_crash(self):
        assert classify_rgb(0, 0, 0) in NLCD_CLASSES
        assert classify_rgb(255, 255, 255) in NLCD_CLASSES


# ---------------------------------------------------------------------------
# tactical_profile — doctrine
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTacticalProfile:
    def test_keys_present(self):
        prof = tactical_profile(42)
        assert set(prof) == {"cover", "concealment", "mobility_cost",
                             "passable", "name", "category"}

    def test_water_is_impassable(self):
        prof = tactical_profile(11)
        assert prof["passable"] is False
        assert prof["mobility_cost"] == 999.0
        assert prof["cover"] == 0.0 and prof["concealment"] == 0.0

    def test_forest_high_concealment_low_cover(self):
        assert tactical_profile(42)["concealment"] == 0.9   # evergreen densest
        assert tactical_profile(41)["concealment"] == 0.85
        assert tactical_profile(43)["concealment"] == 0.85
        for code in (41, 42, 43):
            assert tactical_profile(code)["cover"] == 0.2
            assert tactical_profile(code)["mobility_cost"] == 3.0

    def test_developed_high_cover(self):
        assert tactical_profile(24)["cover"] == 0.9
        assert tactical_profile(23)["cover"] == 0.75
        assert tactical_profile(24)["concealment"] == 0.8

    def test_wetland_mobility_penalty(self):
        assert tactical_profile(90)["mobility_cost"] == 4.0
        assert tactical_profile(95)["mobility_cost"] == 6.0
        assert tactical_profile(90)["passable"] is True

    def test_open_ground_cheap_and_exposed(self):
        # Herbaceous / pasture: no cover, minimal concealment, near-free travel.
        for code in (71, 81):
            prof = tactical_profile(code)
            assert prof["cover"] == 0.0
            assert prof["concealment"] == 0.2
            assert prof["mobility_cost"] == 1.1

    def test_unknown_code_neutral(self):
        assert tactical_profile(9999) == {
            "cover": 0.0, "concealment": 0.0, "mobility_cost": 1.0,
            "passable": True, "name": "Unknown", "category": "unknown",
        }

    def test_none_code_neutral(self):
        assert tactical_profile(None)["category"] == "unknown"


# ---------------------------------------------------------------------------
# tactical_profile — SEASONAL concealment (foliage modifier)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSeasonalConcealment:
    """Forest/shrub concealment scales with the environment foliage factor.

    Ties the GIS land-cover pillar to the weather+seasons pillar: a bare winter
    forest gives little concealment; a summer forest gives full concealment.
    ``foliage=1.0`` (default) is byte-identical to the static doctrine so the
    costmap and every golden are unaffected.
    """

    def test_default_foliage_is_byte_identical(self):
        # No foliage arg and foliage=1.0 must both equal the static table.
        for code in NLCD_CLASSES:
            base = tactical_profile(code)
            assert tactical_profile(code, foliage=1.0) == base
            # foliage >= 1.0 never boosts above full canopy.
            assert tactical_profile(code, foliage=1.5)["concealment"] == \
                base["concealment"]

    def test_forest_concealment_scales_with_foliage(self):
        # Evergreen forest base concealment 0.9.
        base = tactical_profile(42)["concealment"]
        assert base == 0.9
        # Summer-lush ~0.9 canopy -> ~0.81; bare-winter ~0.13 -> ~0.117.
        summer = tactical_profile(42, foliage=0.9)["concealment"]
        winter = tactical_profile(42, foliage=0.13)["concealment"]
        assert summer == pytest.approx(0.9 * 0.9)
        assert winter == pytest.approx(0.9 * 0.13)
        assert winter < summer < base

    def test_shrub_concealment_scales_with_foliage(self):
        base = tactical_profile(52)["concealment"]  # Shrub/Scrub 0.5
        assert base == 0.5
        assert tactical_profile(52, foliage=0.2)["concealment"] == pytest.approx(0.1)
        assert tactical_profile(51, foliage=0.2)["concealment"] == pytest.approx(0.1)

    def test_foliage_never_touches_cover_mobility_passable(self):
        # The costmap reads these; they must be season-invariant.
        for code in (41, 42, 43, 51, 52):
            bare = tactical_profile(code, foliage=0.05)
            full = tactical_profile(code)
            assert bare["cover"] == full["cover"]
            assert bare["mobility_cost"] == full["mobility_cost"]
            assert bare["passable"] == full["passable"]
            assert bare["category"] == full["category"]

    def test_non_vegetation_unaffected_by_foliage(self):
        # Water, developed, barren, snow, herbaceous, cultivated, wetland keep a
        # STATIC concealment regardless of the season.
        for code in (11, 12, 21, 22, 23, 24, 31, 71, 72, 81, 82, 90, 95):
            base = tactical_profile(code)["concealment"]
            for f in (0.05, 0.5, 0.9):
                assert tactical_profile(code, foliage=f)["concealment"] == base

    def test_forest_summer_vs_winter_via_seasonal_cycle(self):
        # Drive the real environment foliage_state at a temperate latitude.
        from tritium_lib.sim_engine.environment import SeasonalCycle
        lat = 40.0  # Boulder foothills
        summer = SeasonalCycle(day_of_year=172, latitude=lat).foliage_state()   # ~Jun 21
        winter = SeasonalCycle(day_of_year=355, latitude=lat).foliage_state()   # ~Dec 21
        assert winter < summer                       # bare winter, lush summer
        c_summer = tactical_profile(42, foliage=summer)["concealment"]
        c_winter = tactical_profile(42, foliage=winter)["concealment"]
        assert c_winter < c_summer
        # Winter forest gives markedly LESS concealment than summer.
        assert c_winter < 0.5 * c_summer

    def test_evergreen_tropical_unchanged_between_seasons(self):
        # Latitude-damped foliage keeps the tropics ~evergreen: forest
        # concealment barely swings near the equator but swings hard at
        # temperate latitude.  At the equator it is EXACTLY constant.
        from tritium_lib.sim_engine.environment import SeasonalCycle

        def swing(lat):
            s = SeasonalCycle(day_of_year=172, latitude=lat).foliage_state()
            w = SeasonalCycle(day_of_year=355, latitude=lat).foliage_state()
            cs = tactical_profile(42, foliage=s)["concealment"]
            cw = tactical_profile(42, foliage=w)["concealment"]
            return cs - cw

        # Equator: evergreen, zero seasonal swing.
        assert swing(0.0) == pytest.approx(0.0, abs=1e-9)
        # Tropical swing << temperate swing (Boulder 40deg).
        assert swing(5.0) < 0.3 * swing(40.0)

    def test_foliage_clamped_below_zero(self):
        # Defensive: a negative foliage clamps to 0 (fully bare), never negative.
        assert tactical_profile(42, foliage=-1.0)["concealment"] == 0.0


# ---------------------------------------------------------------------------
# LandCoverGrid — geometry, round-trip, rollups
# ---------------------------------------------------------------------------

def _grid():
    # 4x3 grid, row 0 = north; one NoData cell.
    codes = [42, 42, 41, None,
             23, 23, 71, 71,
             11, 11, 82, 82]
    return LandCoverGrid(west=-121.912, south=37.704, east=-121.880, north=37.728,
                         ncols=4, nrows=3, codes=codes, source="nlcd")


@pytest.mark.unit
class TestLandCoverGrid:
    def test_geometry_matches_elevation_grid(self):
        lc = _grid()
        ev = ElevationGrid(west=lc.west, south=lc.south, east=lc.east,
                           north=lc.north, ncols=lc.ncols, nrows=lc.nrows,
                           values=[0.0] * (lc.ncols * lc.nrows))
        for ix in range(lc.ncols):
            assert lc.cell_lon(ix) == ev.cell_lon(ix)
        for iy in range(lc.nrows):
            assert lc.cell_lat(iy) == ev.cell_lat(iy)

    def test_row0_is_north(self):
        lc = _grid()
        assert lc.cell_lat(0) == pytest.approx(lc.north)
        assert lc.cell_lat(lc.nrows - 1) == pytest.approx(lc.south)

    def test_code_at(self):
        lc = _grid()
        assert lc.code_at(0, 0) == 42
        assert lc.code_at(3, 0) is None
        assert lc.code_at(2, 2) == 82

    def test_code_at_out_of_range(self):
        with pytest.raises(IndexError):
            _grid().code_at(4, 0)

    def test_round_trip(self):
        lc = _grid()
        d = lc.to_dict()
        assert set(d) == {"west", "south", "east", "north",
                          "ncols", "nrows", "codes", "source"}
        rebuilt = LandCoverGrid.from_dict(d)
        assert rebuilt.codes == lc.codes
        assert rebuilt.source == lc.source
        assert rebuilt.to_dict() == d

    def test_from_dict_tolerates_fixture_markers(self):
        d = _grid().to_dict()
        d["fixture"] = True
        d["bbox"] = [d["west"], d["south"], d["east"], d["north"]]
        rebuilt = LandCoverGrid.from_dict(d)
        assert rebuilt.ncols == 4 and rebuilt.nrows == 3

    def test_dominant_category(self):
        # forest x3, developed x2, herbaceous x2, water x2, cultivated x2 -> forest.
        assert _grid().dominant_category() == "forest"

    def test_dominant_category_all_nodata_is_none(self):
        lc = LandCoverGrid(west=0, south=0, east=1, north=1, ncols=2, nrows=2,
                          codes=[None, None, None, None])
        assert lc.dominant_category() is None

    def test_tactical_field(self):
        lc = _grid()
        field = lc.tactical_field()
        assert len(field) == lc.ncols * lc.nrows
        assert field[0]["category"] == "forest"
        assert field[3]["category"] == "unknown"  # NoData -> neutral
        assert field[8]["passable"] is False       # water cell

    def test_tactical_field_default_byte_identical(self):
        lc = _grid()
        assert lc.tactical_field(foliage=1.0) == lc.tactical_field()

    def test_tactical_field_seasonal_thins_forest_only(self):
        lc = _grid()
        summer = lc.tactical_field()               # foliage 1.0
        winter = lc.tactical_field(foliage=0.13)
        # Forest cell 0 concealment drops; costmap keys unchanged.
        assert winter[0]["concealment"] < summer[0]["concealment"]
        assert winter[0]["mobility_cost"] == summer[0]["mobility_cost"]
        assert winter[0]["passable"] == summer[0]["passable"]
        # Developed cell 4 (code 23) is unaffected.
        assert winter[4]["concealment"] == summer[4]["concealment"]
        # Water cell 8 (code 11) unaffected + still impassable.
        assert winter[8]["concealment"] == summer[8]["concealment"]
        assert winter[8]["passable"] is False

    def test_concealment_at_samples_nearest_cell_and_scales(self):
        lc = _grid()
        # Cell (0,0) is evergreen forest (42), near the NW corner.
        c_summer = lc.concealment_at(lc.north, lc.west, foliage=1.0)
        c_winter = lc.concealment_at(lc.north, lc.west, foliage=0.13)
        assert c_summer == pytest.approx(0.9)
        assert c_winter == pytest.approx(0.9 * 0.13)
        assert c_winter < c_summer
        # A developed cell (row 1) is season-invariant.
        dev_lat = lc.cell_lat(1)
        assert lc.concealment_at(dev_lat, lc.west, foliage=0.13) == \
            lc.concealment_at(dev_lat, lc.west, foliage=1.0)

    def test_concealment_at_out_of_grid_is_exposed(self):
        lc = _grid()
        # Far outside the bbox clamps to an edge cell (never raises); a degenerate
        # empty grid returns 0.0.
        empty = LandCoverGrid(west=0, south=0, east=1, north=1, ncols=0, nrows=0)
        assert empty.concealment_at(0.5, 0.5) == 0.0

    def test_mean_concealment_thins_in_winter(self):
        lc = _grid()
        summer = lc.mean_concealment(foliage=1.0)
        winter = lc.mean_concealment(foliage=0.13)
        assert winter < summer
        # Default equals summer (byte-identical path).
        assert lc.mean_concealment() == summer


# ---------------------------------------------------------------------------
# Fetcher — URL builder + degradation chain (no network)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFetcherUrl:
    def test_wms_url_has_mercator_bbox(self):
        f = NlcdLandCoverFetcher()
        url = f._build_url(DUBLIN, 32, 32)
        assert "request=GetMap" in url
        assert "width=32" in url and "height=32" in url
        assert "NLCD_2021_Land_Cover_L48" in url
        # EPSG:3857 (colon percent-encoded by urlencode) + a mercator x (~ -1.35e7).
        assert "EPSG%3A3857" in url
        from tritium_lib.geo.gis.fetchers import lonlat_to_web_mercator
        minx, _ = lonlat_to_web_mercator(DUBLIN.west, DUBLIN.south)
        assert str(minx) in url


@pytest.mark.unit
class TestFetcherDegradation:
    """Force the live step to fail so only cache/fixture/empty run (no network)."""

    @pytest.fixture(autouse=True)
    def _no_network(self, monkeypatch):
        def _boom(url, timeout=0):
            raise OSError("offline (test)")
        monkeypatch.setattr(fetchers_mod, "_http_bytes", _boom)

    def test_dublin_bbox_returns_dublin_fixture(self):
        f = NlcdLandCoverFetcher(cache=None)
        grid = f.fetch_grid(DUBLIN, 32, 32)
        assert grid.source == "nlcd-fixture"
        assert grid.ncols == 32 and grid.nrows == 32
        # Real captured Dublin data — mostly developed, some open ground.
        assert grid.dominant_category() == "developed"
        assert any(c is not None for c in grid.codes)

    def test_boulder_bbox_returns_boulder_fixture(self):
        f = NlcdLandCoverFetcher(cache=None)
        grid = f.fetch_grid(BOULDER, 32, 32)
        assert grid.source == "nlcd-fixture"
        # Boulder has real forest at elevation (evergreen present).
        cats = {tactical_profile(c)["category"] for c in grid.codes if c is not None}
        assert "forest" in cats
        assert "developed" in cats

    def test_far_bbox_degrades_to_empty(self):
        f = NlcdLandCoverFetcher(cache=None)
        grid = f.fetch_grid(GeoBBox(2.0, 48.0, 2.1, 48.1), 8, 8)  # Paris
        assert grid.source == "nlcd-empty"
        assert grid.codes == [None] * 64

    def test_cache_hit_before_fixture(self, tmp_path):
        from tritium_lib.geo.gis import GISCache
        cache = GISCache(tmp_path)
        f = NlcdLandCoverFetcher(cache=cache)
        # Seed the cache with a grid dict for the exact key.
        seeded = LandCoverGrid(west=DUBLIN.west, south=DUBLIN.south,
                               east=DUBLIN.east, north=DUBLIN.north,
                               ncols=4, nrows=4, codes=[71] * 16,
                               source="nlcd-cached")
        key = cache.key(f.SOURCE, DUBLIN, ncols=4, nrows=4)
        cache.put(key, seeded.to_dict())
        grid = f.fetch_grid(DUBLIN, 4, 4)
        assert grid.source == "nlcd-cached"
        assert grid.codes == [71] * 16


# ---------------------------------------------------------------------------
# parse_png_grid — needs Pillow (skip cleanly when absent, keeping lib PIL-free)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParsePngGrid:
    def _png(self, size, pixels):
        Image = pytest.importorskip("PIL.Image")
        im = Image.new("RGBA", size, (0, 0, 0, 0))
        for (ix, iy), rgba in pixels.items():
            im.putpixel((ix, iy), rgba)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def test_classifies_each_cell_and_marks_transparent_nodata(self):
        pytest.importorskip("PIL")
        png = self._png((2, 2), {
            (0, 0): (28, 95, 44, 255),    # evergreen -> 42
            (1, 0): (237, 0, 0, 255),     # dev medium (WMS shade) -> 23
            (0, 1): (70, 107, 159, 255),  # water -> 11
            (1, 1): (0, 0, 0, 0),         # transparent -> None
        })
        grid = NlcdLandCoverFetcher.parse_png_grid(png, DUBLIN, 2, 2)
        # Row-major, row 0 = north: [(0,0),(1,0),(0,1),(1,1)].
        assert grid.codes == [42, 23, 11, None]
        assert grid.source == "nlcd"
        assert grid.west == DUBLIN.west and grid.north == DUBLIN.north

    def test_resamples_to_requested_grid_size(self):
        pytest.importorskip("PIL")
        # Solid 8x8 evergreen -> any downsample is all 42.
        png = self._png((8, 8), {
            (ix, iy): (28, 95, 44, 255) for ix in range(8) for iy in range(8)
        })
        grid = NlcdLandCoverFetcher.parse_png_grid(png, DUBLIN, 2, 2)
        assert grid.codes == [42, 42, 42, 42]

    def test_live_path_end_to_end_with_synthetic_png(self, monkeypatch):
        pytest.importorskip("PIL")
        png = self._png((2, 2), {
            (ix, iy): (28, 95, 44, 255) for ix in range(2) for iy in range(2)
        })
        monkeypatch.setattr(fetchers_mod, "_http_bytes",
                            lambda url, timeout=0: png)
        f = NlcdLandCoverFetcher(cache=None)
        grid = f.fetch_grid(DUBLIN, 2, 2)
        assert grid.source == "nlcd"        # live path taken, not a fixture
        assert grid.codes == [42, 42, 42, 42]

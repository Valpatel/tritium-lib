# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.gis.capture — the reusable AO fixture capture tool.

NETWORK IS NEVER TOUCHED: every fetcher here is a stub whose ``fetch`` /
``fetch_grid`` returns a canned payload (or raises).  The tool must cap, round,
stamp fixture+bbox markers, and skip-not-raise on a bad source.
"""

import json

import pytest

from tritium_lib.geo.gis.capture import capture_ao_pack
from tritium_lib.geo.gis.models import ElevationGrid, GeoBBox

AO = GeoBBox.from_string("-105.30,39.98,-105.26,40.02")


# ---------------------------------------------------------------------------
# Stub fetchers (dispatch: fetch_grid => DEM, else vector)
# ---------------------------------------------------------------------------
class _StubVector:
    def __init__(self, fc, source="tiger", fixture_name="tiger_roads_ao.json"):
        self._fc = fc
        self.SOURCE = source
        self.FIXTURE_NAME = fixture_name

    def fetch(self, bbox):
        return self._fc


class _RaisingVector:
    SOURCE = "fema"
    FIXTURE_NAME = "fema_flood_ao.json"

    def fetch(self, bbox):
        raise RuntimeError("simulated live failure")


class _StubDem:
    SOURCE = "usgs"
    FIXTURE_NAME = "usgs_dem_ao.json"

    def __init__(self, grid):
        self._grid = grid

    def fetch_grid(self, bbox, ncols=16, nrows=16):
        return self._grid


def _road(lon, lat, kind="S1400"):
    return {
        "type": "Feature",
        "geometry": {"type": "LineString",
                     "coordinates": [[lon, lat], [lon + 0.0001, lat + 0.0001]]},
        "properties": {"source": "tiger", "kind": kind, "name": ""},
    }


def _load(tmp_path, filename):
    return json.loads((tmp_path / filename).read_text(encoding="utf-8"))


class TestCaptureAoPack:
    @pytest.mark.unit
    def test_caps_rounds_and_stamps_markers(self, tmp_path):
        # Five high-precision road features; cap to 2, round to 3 dp.
        fc = {"type": "FeatureCollection", "features": [
            _road(-105.123456789, 39.987654321),
            _road(-105.223456789, 39.997654321),
            _road(-105.323456789, 40.007654321),
            _road(-105.423456789, 40.017654321),
            _road(-105.523456789, 40.027654321),
        ]}
        summary = capture_ao_pack(
            AO, "boulder", tmp_path,
            fetchers=[_StubVector(fc)], max_features=2, precision=3,
        )
        assert summary == {"tiger_roads_boulder.json": 2}
        data = _load(tmp_path, "tiger_roads_boulder.json")
        assert data["type"] == "FeatureCollection"
        assert data["fixture"] is True
        assert data["bbox"] == [-105.3, 39.98, -105.26, 40.02]
        assert len(data["features"]) == 2                      # capped
        # Coordinates rounded to 3 dp.
        assert data["features"][0]["geometry"]["coordinates"][0] == [-105.123, 39.988]

    @pytest.mark.unit
    def test_skips_failing_and_empty_sources(self, tmp_path):
        empty = _StubVector({"type": "FeatureCollection", "features": []},
                            source="noaa", fixture_name="noaa_alerts_ao.json")
        summary = capture_ao_pack(
            AO, "boulder", tmp_path,
            fetchers=[_RaisingVector(), empty],
        )
        assert summary["fema_flood_boulder.json"].startswith("skipped")
        assert summary["noaa_alerts_boulder.json"].startswith("skipped")
        # No files written for skipped sources.
        assert not (tmp_path / "fema_flood_boulder.json").exists()
        assert not (tmp_path / "noaa_alerts_boulder.json").exists()

    @pytest.mark.unit
    def test_dem_capture_marks_fixture_and_rounds_values(self, tmp_path):
        grid = ElevationGrid(
            west=-105.30, south=39.98, east=-105.26, north=40.02,
            ncols=2, nrows=2,
            values=[1650.126789, None, 1720.987654, 1580.5],
            source="usgs", resolution_m=10,
        )
        summary = capture_ao_pack(AO, "boulder", tmp_path, fetchers=[_StubDem(grid)])
        assert summary == {"usgs_dem_boulder.json": "2x2"}
        data = _load(tmp_path, "usgs_dem_boulder.json")
        assert data["fixture"] is True
        assert data["bbox"] == [-105.3, 39.98, -105.26, 40.02]
        assert data["source"] == "usgs-fixture"            # marked as a fixture
        assert data["values"] == [1650.13, None, 1720.99, 1580.5]  # 2 dp, NoData kept
        # Round-trips through ElevationGrid.from_dict.
        assert ElevationGrid.from_dict(data).ncols == 2

    @pytest.mark.unit
    def test_dem_all_nodata_is_skipped(self, tmp_path):
        grid = ElevationGrid(
            west=-105.30, south=39.98, east=-105.26, north=40.02,
            ncols=2, nrows=2, values=[None, None, None, None],
            source="usgs-empty",
        )
        summary = capture_ao_pack(AO, "boulder", tmp_path, fetchers=[_StubDem(grid)])
        assert summary["usgs_dem_boulder.json"].startswith("skipped")
        assert not (tmp_path / "usgs_dem_boulder.json").exists()

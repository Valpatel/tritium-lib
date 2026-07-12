# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.gis.ao_packs — the AO pack registry.

Every field is derived from the packaged fixtures, so these tests double as a
guard that the two real packs (Dublin CA, Boulder CO) stay coherent with the
fixture files on disk (bbox markers, DEM grids, per-layer presence).
"""

import json
from importlib import resources

import pytest

from tritium_lib.geo.gis import (
    AOPack,
    active_ao_id,
    get_ao_pack,
    list_ao_packs,
)
from tritium_lib.geo.gis import ao_packs as ao_mod

_FIXTURE_PKG = "tritium_lib.geo.gis.fixtures"

# Canonical AO boxes (from the fixture docstring / capture) — used only to
# ASSERT the registry derived the same numbers, never to feed it.
DUBLIN_BBOX = (-121.912, 37.704, -121.880, 37.728)
BOULDER_BBOX = (-105.30, 39.98, -105.26, 40.02)


def _fixture_bbox(name: str):
    """Read a fixture's own declared bbox straight off disk for cross-check."""
    resource = resources.files(_FIXTURE_PKG).joinpath(name)
    data = json.loads(resource.read_text(encoding="utf-8"))
    marker = data.get("bbox")
    if isinstance(marker, (list, tuple)) and len(marker) == 4:
        return tuple(float(v) for v in marker)
    return (
        float(data["west"]), float(data["south"]),
        float(data["east"]), float(data["north"]),
    )


@pytest.mark.unit
class TestListAOPacks:
    def test_lists_both_real_packs(self):
        packs = list_ao_packs()
        ids = [p.id for p in packs]
        assert "dublin" in ids
        assert "boulder" in ids
        assert all(isinstance(p, AOPack) for p in packs)

    def test_registry_order_dublin_first(self):
        # Dublin is the original demo AO — it leads the picker.
        assert [p.id for p in list_ao_packs()][:2] == ["dublin", "boulder"]

    def test_dublin_bbox_matches_fixture(self):
        pack = get_ao_pack("dublin")
        assert pack is not None
        assert pack.bbox == pytest.approx(DUBLIN_BBOX)
        # ...and matches what the DEM fixture itself declares.
        assert pack.bbox == pytest.approx(_fixture_bbox("usgs_dem_ao.json"))

    def test_boulder_bbox_matches_fixture(self):
        pack = get_ao_pack("boulder")
        assert pack is not None
        assert pack.bbox == pytest.approx(BOULDER_BBOX)
        assert pack.bbox == pytest.approx(_fixture_bbox("usgs_dem_boulder.json"))

    def test_centers_inside_their_bbox(self):
        for pack in list_ao_packs():
            w, s, e, n = pack.bbox
            assert w < pack.center_lng < e
            assert s < pack.center_lat < n
            # center() is the geometric midpoint
            assert pack.center_lng == pytest.approx((w + e) / 2.0)
            assert pack.center_lat == pytest.approx((s + n) / 2.0)

    def test_boulder_has_no_noaa_layer(self):
        # Legitimately-empty layer: no active NWS alerts at capture time.
        boulder = get_ao_pack("boulder")
        assert "noaa_alerts" not in boulder.layers
        assert "usgs_dem" in boulder.layers
        assert "tiger_roads" in boulder.layers
        assert "fema_flood" in boulder.layers
        assert "osm_buildings" in boulder.layers

    def test_dublin_has_noaa_layer(self):
        dublin = get_ao_pack("dublin")
        assert "noaa_alerts" in dublin.layers
        assert "usgs_dem" in dublin.layers

    def test_reports_nlcd_and_nhd_layers_when_shipped(self):
        # Regression: the registry must report EVERY packaged layer, not just
        # the original five.  Dev added NLCD land cover + NHD hydrography
        # fixtures for both AOs; a stale _LAYER_STEMS silently dropped them and
        # the AO switcher under-reported each battlefield.
        for pack in (get_ao_pack("dublin"), get_ao_pack("boulder")):
            assert "nlcd" in pack.layers, f"{pack.id} missing nlcd"
            assert "nhd_hydro" in pack.layers, f"{pack.id} missing nhd_hydro"

    def test_layers_are_registry_ordered(self):
        # layers preserve _LAYER_STEMS (sensor-priority) order for a stable UI.
        from tritium_lib.geo.gis.ao_packs import _LAYER_STEMS
        for pack in list_ao_packs():
            idxs = [_LAYER_STEMS.index(s) for s in pack.layers]
            assert idxs == sorted(idxs)


@pytest.mark.unit
class TestActiveAndContainment:
    def test_contains_own_center(self):
        for pack in list_ao_packs():
            assert pack.contains(pack.center_lat, pack.center_lng)

    def test_active_ao_from_center(self):
        dublin = get_ao_pack("dublin")
        boulder = get_ao_pack("boulder")
        assert active_ao_id(dublin.center_lat, dublin.center_lng) == "dublin"
        assert active_ao_id(boulder.center_lat, boulder.center_lng) == "boulder"

    def test_active_ao_none_when_far_away(self):
        # Mid-Atlantic — inside no packaged AO.
        assert active_ao_id(0.0, -30.0) is None

    def test_boulder_center_not_in_dublin(self):
        boulder = get_ao_pack("boulder")
        dublin = get_ao_pack("dublin")
        assert not dublin.contains(boulder.center_lat, boulder.center_lng)


@pytest.mark.unit
class TestSerializationAndDegradation:
    def test_to_dict_shape(self):
        d = get_ao_pack("boulder").to_dict()
        assert d["id"] == "boulder"
        assert d["name"] == "Boulder, CO"
        assert d["bbox"] == pytest.approx(list(BOULDER_BBOX))
        assert set(d["center"]) == {"lat", "lng"}
        assert isinstance(d["layers"], list)
        assert "noaa_alerts" not in d["layers"]

    def test_get_unknown_pack_is_none(self):
        assert get_ao_pack("atlantis") is None

    def test_empty_when_fixtures_absent(self, monkeypatch):
        # Simulate a build with no packaged fixtures at all: every fixture
        # lookup misses -> no pack has data -> empty list (picker hides).
        monkeypatch.setattr(ao_mod, "_fixture_exists", lambda name: False)
        assert list_ao_packs() == []

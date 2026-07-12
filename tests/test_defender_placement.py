# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Key-terrain defender placement (tritium_lib.mission.defense).

Proves the helper garrisons the tactically real crossings: highest hold_value
first, nearest defender to each, deterministic, and driven by the SAME
chokepoint tactical-object contract the GIS layer projects (built here via
``chokepoint_tactical_object`` so the fixture is authentic, not hand-waved).
The module under test imports no GIS code — the fixture does, to prove the
contract lines up.
"""

from __future__ import annotations

from tritium_lib.geo.gis.chokepoints import chokepoint_tactical_object
from tritium_lib.mission import (
    assign_defenders_to_chokepoints,
    rank_hold_points,
)


def _feature(cid, lon, lat, road_name, road_kind, water_name, water_kind):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "source": "chokepoint",
            "id": cid,
            "kind": "bridge",
            "name": f"{road_name} @ {water_name}",
            "road_name": road_name,
            "road_kind": road_kind,
            "water_name": water_name,
            "water_kind": water_kind,
        },
    }


def _tac(cid, lon, lat, road_name, road_kind, water_name, water_kind):
    """A real chokepoint tactical object (hold_value/sever/tags computed)."""
    return chokepoint_tactical_object(
        _feature(cid, lon, lat, road_name, road_kind, water_name, water_kind)
    )


# A little canyon town: a primary-road river bridge (key terrain), a local-road
# stream bridge, and a trail ditch culvert (throwaway).
def _boulder_chokepoints():
    return [
        _tac("chk_hwy", -105.30, 40.015, "US-36", "S1100", "Boulder Creek", "river"),
        _tac("chk_9th", -105.28, 40.017, "9th St", "S1400", "Gregory Creek", "stream"),
        _tac("chk_trail", -105.29, 40.020, "Foothills Trail", "S1710", "Ditch A", "canal"),
    ]


class TestRankHoldPoints:
    def test_orders_by_hold_value_descending(self):
        ranked = rank_hold_points(_boulder_chokepoints())
        ids = [c["id"] for c in ranked]
        # US-36 river bridge (hold 3+2+2=7) > 9th St stream bridge (3+1+1=5)
        # > trail ditch culvert (3-1=2).
        assert ids == ["chk_hwy", "chk_9th", "chk_trail"]
        assert ranked[0]["hold_value"] == 7
        assert ranked[0]["tags"].count("key_terrain") == 1

    def test_min_hold_filters(self):
        ranked = rank_hold_points(_boulder_chokepoints(), min_hold=5)
        assert [c["id"] for c in ranked] == ["chk_hwy", "chk_9th"]

    def test_deterministic_tiebreak_on_id(self):
        # Two equal-value crossings -> stable id ordering regardless of input order.
        a = _tac("chk_bbb", -105.0, 40.0, "B St", "S1400", "Creek", "stream")
        b = _tac("chk_aaa", -105.1, 40.1, "A St", "S1400", "Creek", "stream")
        assert a["hold_value"] == b["hold_value"]
        assert [c["id"] for c in rank_hold_points([a, b])] == ["chk_aaa", "chk_bbb"]
        assert [c["id"] for c in rank_hold_points([b, a])] == ["chk_aaa", "chk_bbb"]

    def test_empty_input(self):
        assert rank_hold_points([]) == []
        assert rank_hold_points([{"no": "position"}]) == []


class TestAssignDefenders:
    def test_scarce_defenders_go_to_top_holdpoints(self):
        cps = _boulder_chokepoints()
        # Two defenders, three crossings -> the two best crossings get manned,
        # the throwaway culvert does not.
        defenders = [
            {"id": "rover-1", "position": {"lon": -105.30, "lat": 40.014}},
            {"id": "rover-2", "position": {"lon": -105.28, "lat": 40.018}},
        ]
        out = assign_defenders_to_chokepoints(cps, defenders)
        assert len(out) == 2
        held = {a["chokepoint_id"] for a in out}
        assert held == {"chk_hwy", "chk_9th"}
        assert "chk_trail" not in held
        # Highest-value crossing is listed first and is key terrain.
        assert out[0]["chokepoint_id"] == "chk_hwy"
        assert out[0]["hold_value"] == 7
        assert out[0]["key_terrain"] is True
        assert out[0]["sever"] is True

    def test_nearest_defender_assigned_to_each(self):
        cps = _boulder_chokepoints()
        # rover-2 sits basically on the highway bridge; rover-1 by 9th St.
        defenders = [
            {"id": "rover-1", "position": {"lon": -105.281, "lat": 40.017}},
            {"id": "rover-2", "position": {"lon": -105.301, "lat": 40.015}},
        ]
        out = assign_defenders_to_chokepoints(cps, defenders)
        by_cp = {a["chokepoint_id"]: a["defender_id"] for a in out}
        assert by_cp["chk_hwy"] == "rover-2"   # closest to the highway bridge
        assert by_cp["chk_9th"] == "rover-1"   # closest to the 9th St bridge
        # Distances are reported and small (defenders were placed nearby).
        assert all(a["distance_m"] < 500 for a in out)

    def test_max_per_point_stacks_top_crossing(self):
        cps = _boulder_chokepoints()
        defenders = [
            {"id": "a", "position": {"lon": -105.30, "lat": 40.0}},
            {"id": "b", "position": {"lon": -105.30, "lat": 40.0}},
            {"id": "c", "position": {"lon": -105.30, "lat": 40.0}},
        ]
        out = assign_defenders_to_chokepoints(cps, defenders, max_per_point=2)
        counts: dict[str, int] = {}
        for a in out:
            counts[a["chokepoint_id"]] = counts.get(a["chokepoint_id"], 0) + 1
        assert counts["chk_hwy"] == 2   # top crossing double-manned first
        assert counts["chk_9th"] == 1   # third defender to the next crossing

    def test_deterministic(self):
        cps = _boulder_chokepoints()
        defenders = [
            {"id": "x", "position": {"lon": -105.31, "lat": 40.01}},
            {"id": "y", "position": {"lon": -105.27, "lat": 40.02}},
            {"id": "z", "position": {"lon": -105.29, "lat": 40.015}},
        ]
        a = assign_defenders_to_chokepoints(cps, defenders)
        b = assign_defenders_to_chokepoints(cps, defenders)
        assert a == b

    def test_flexible_position_shapes(self):
        cps = _boulder_chokepoints()
        # {lon,lat}, [lon,lat], and target_id instead of id all resolve.
        defenders = [
            {"id": "d1", "lon": -105.30, "lat": 40.015},
            {"id": "d2", "position": [-105.28, 40.017]},
            {"target_id": "d3", "position": {"lng": -105.29, "lat": 40.02}},
        ]
        out = assign_defenders_to_chokepoints(cps, defenders)
        assert len(out) == 3
        assert {a["defender_id"] for a in out} == {"d1", "d2", "d3"}

    def test_no_defenders_or_no_chokepoints(self):
        cps = _boulder_chokepoints()
        assert assign_defenders_to_chokepoints(cps, []) == []
        assert assign_defenders_to_chokepoints([], [{"id": "d", "lon": 0, "lat": 0}]) == []

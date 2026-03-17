# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the diplomacy and faction system."""

from __future__ import annotations

import pytest

from tritium_lib.sim_engine.factions import (
    DiplomacyEngine,
    DiplomaticRelation,
    Faction,
    Relation,
    FACTION_PRESETS,
    load_preset,
    _centroid,
    _relation_color,
    _RELATION_ORDER,
    _ORDER_TO_RELATION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gov() -> Faction:
    return Faction(faction_id="gov", name="Government", color="#00ff00",
                   ideology="government", strength=0.8, wealth=0.7)

def _reb() -> Faction:
    return Faction(faction_id="reb", name="Rebels", color="#ff0000",
                   ideology="rebel", strength=0.4, wealth=0.2)

def _civ() -> Faction:
    return Faction(faction_id="civ", name="Civilians", color="#ffff00",
                   ideology="civilian", strength=0.05, wealth=0.3, morale=0.5)

def _merc() -> Faction:
    return Faction(faction_id="merc", name="Mercenaries", color="#ff8800",
                   ideology="mercenary", strength=0.6, wealth=0.5)

def _engine_2() -> DiplomacyEngine:
    """Engine with gov + reb."""
    e = DiplomacyEngine()
    e.add_faction(_gov())
    e.add_faction(_reb())
    return e

def _engine_3() -> DiplomacyEngine:
    """Engine with gov + reb + civ."""
    e = _engine_2()
    e.add_faction(_civ())
    return e


# ---------------------------------------------------------------------------
# Relation enum
# ---------------------------------------------------------------------------

class TestRelationEnum:
    def test_values(self):
        assert Relation.ALLIED.value == "allied"
        assert Relation.WAR.value == "war"

    def test_all_relations_in_order(self):
        assert len(_RELATION_ORDER) == 6

    def test_order_round_trip(self):
        for r in Relation:
            assert _ORDER_TO_RELATION[_RELATION_ORDER[r]] is r


# ---------------------------------------------------------------------------
# Faction dataclass
# ---------------------------------------------------------------------------

class TestFaction:
    def test_defaults(self):
        f = Faction(faction_id="x", name="X", color="#fff", ideology="rebel",
                     strength=0.5, wealth=0.5)
        assert f.morale == 0.7
        assert f.territory == []
        assert f.leader is None

    def test_with_territory(self):
        f = Faction(faction_id="a", name="A", color="#000", ideology="government",
                     strength=1.0, wealth=1.0,
                     territory=[(0.0, 0.0), (10.0, 10.0)])
        assert len(f.territory) == 2

    def test_with_leader(self):
        f = Faction(faction_id="a", name="A", color="#000", ideology="mercenary",
                     strength=0.5, wealth=0.5, leader="unit_42")
        assert f.leader == "unit_42"


# ---------------------------------------------------------------------------
# DiplomaticRelation dataclass
# ---------------------------------------------------------------------------

class TestDiplomaticRelation:
    def test_defaults(self):
        dr = DiplomaticRelation(faction_a="a", faction_b="b")
        assert dr.relation == Relation.NEUTRAL
        assert dr.trust == 0.5
        assert dr.trade_active is False
        assert dr.ceasefire_until is None
        assert dr.history == []

    def test_log(self):
        dr = DiplomaticRelation(faction_a="a", faction_b="b")
        dr._log("test", "something happened", 10.5)
        assert len(dr.history) == 1
        assert dr.history[0]["type"] == "test"
        assert dr.history[0]["sim_time"] == 10.5

    def test_log_captures_state(self):
        dr = DiplomaticRelation(faction_a="a", faction_b="b",
                                 relation=Relation.HOSTILE, trust=0.2)
        dr._log("check", "snapshot")
        assert dr.history[0]["relation"] == "hostile"
        assert dr.history[0]["trust"] == 0.2


# ---------------------------------------------------------------------------
# DiplomacyEngine — add / remove factions
# ---------------------------------------------------------------------------

class TestEngineFactions:
    def test_add_faction(self):
        e = DiplomacyEngine()
        e.add_faction(_gov())
        assert "gov" in e.factions

    def test_add_creates_relations(self):
        e = _engine_2()
        assert len(e.relations) == 1  # gov-reb pair

    def test_add_three_creates_three_relations(self):
        e = _engine_3()
        assert len(e.relations) == 3  # gov-reb, civ-gov, civ-reb

    def test_remove_faction(self):
        e = _engine_3()
        e.remove_faction("civ")
        assert "civ" not in e.factions
        assert len(e.relations) == 1  # only gov-reb remains

    def test_remove_nonexistent(self):
        e = _engine_2()
        e.remove_faction("nope")  # should not raise
        assert len(e.factions) == 2

    def test_leader_assignment(self):
        f = _gov()
        f.leader = "unit_cmd"
        e = DiplomacyEngine()
        e.add_faction(f)
        assert e.faction_for_unit("unit_cmd") == "gov"


# ---------------------------------------------------------------------------
# Unit assignment
# ---------------------------------------------------------------------------

class TestUnitAssignment:
    def test_assign_unit(self):
        e = _engine_2()
        e.assign_unit("soldier_1", "gov")
        assert e.faction_for_unit("soldier_1") == "gov"

    def test_unknown_unit(self):
        e = _engine_2()
        assert e.faction_for_unit("nobody") is None

    def test_reassign_unit(self):
        e = _engine_2()
        e.assign_unit("defector", "gov")
        e.assign_unit("defector", "reb")
        assert e.faction_for_unit("defector") == "reb"

    def test_remove_faction_clears_units(self):
        e = _engine_2()
        e.assign_unit("s1", "reb")
        e.remove_faction("reb")
        assert e.faction_for_unit("s1") is None


# ---------------------------------------------------------------------------
# Relation lookup / set
# ---------------------------------------------------------------------------

class TestRelationLookup:
    def test_default_is_neutral(self):
        e = _engine_2()
        dr = e.get_relation("gov", "reb")
        assert dr.relation == Relation.NEUTRAL

    def test_symmetric_key(self):
        e = _engine_2()
        dr1 = e.get_relation("gov", "reb")
        dr2 = e.get_relation("reb", "gov")
        assert dr1 is dr2

    def test_set_relation(self):
        e = _engine_2()
        e.set_relation("gov", "reb", Relation.HOSTILE)
        assert e.get_relation("gov", "reb").relation == Relation.HOSTILE

    def test_set_relation_logs_history(self):
        e = _engine_2()
        e.set_relation("gov", "reb", Relation.SUSPICIOUS)
        dr = e.get_relation("gov", "reb")
        assert len(dr.history) == 1
        assert "neutral -> suspicious" in dr.history[0]["detail"]

    def test_set_war_cancels_trade(self):
        e = _engine_2()
        dr = e.get_relation("gov", "reb")
        dr.trade_active = True
        e.set_relation("gov", "reb", Relation.WAR)
        assert dr.trade_active is False

    def test_get_relation_auto_creates(self):
        e = DiplomacyEngine()
        e.add_faction(_gov())
        e.add_faction(_merc())
        # Force a lookup for a pair that wasn't auto-created
        # (it actually is auto-created, but test the fallback path)
        dr = e.get_relation("gov", "merc")
        assert dr.relation == Relation.NEUTRAL


# ---------------------------------------------------------------------------
# War / ceasefire / alliance
# ---------------------------------------------------------------------------

class TestWarAndPeace:
    def test_declare_war(self):
        e = _engine_2()
        e.declare_war("gov", "reb")
        dr = e.get_relation("gov", "reb")
        assert dr.relation == Relation.WAR
        assert dr.trust <= 0.1
        assert dr.trade_active is False

    def test_declare_war_clears_ceasefire(self):
        e = _engine_2()
        e.propose_ceasefire("gov", "reb", 100.0)
        e.declare_war("gov", "reb")
        assert e.get_relation("gov", "reb").ceasefire_until is None

    def test_propose_ceasefire_accepted(self):
        e = _engine_2()
        e.declare_war("gov", "reb")
        # Trust is ~0.1, set it higher for acceptance
        e.get_relation("gov", "reb").trust = 0.3
        ok = e.propose_ceasefire("gov", "reb", 60.0)
        assert ok is True
        dr = e.get_relation("gov", "reb")
        assert dr.ceasefire_until == 60.0  # sim_time is 0
        assert dr.relation == Relation.HOSTILE  # still hostile, just ceasefire

    def test_propose_ceasefire_rejected_low_trust(self):
        e = _engine_2()
        e.declare_war("gov", "reb")
        e.get_relation("gov", "reb").trust = 0.05
        ok = e.propose_ceasefire("gov", "reb", 60.0)
        assert ok is False

    def test_form_alliance(self):
        e = _engine_2()
        e.form_alliance("gov", "reb")
        dr = e.get_relation("gov", "reb")
        assert dr.relation == Relation.ALLIED
        assert dr.trust >= 0.8
        assert dr.trade_active is True

    def test_break_alliance(self):
        e = _engine_2()
        e.form_alliance("gov", "reb")
        e.break_alliance("gov", "reb")
        dr = e.get_relation("gov", "reb")
        assert dr.relation == Relation.SUSPICIOUS
        assert dr.trust < 0.5
        assert dr.trade_active is False

    def test_break_non_alliance_is_noop(self):
        e = _engine_2()
        e.set_relation("gov", "reb", Relation.FRIENDLY)
        e.break_alliance("gov", "reb")
        # Should not change since they weren't allied
        assert e.get_relation("gov", "reb").relation == Relation.FRIENDLY

    def test_are_hostile(self):
        e = _engine_2()
        assert e.are_hostile("gov", "reb") is False
        e.set_relation("gov", "reb", Relation.HOSTILE)
        assert e.are_hostile("gov", "reb") is True
        e.declare_war("gov", "reb")
        assert e.are_hostile("gov", "reb") is True

    def test_are_allied(self):
        e = _engine_2()
        assert e.are_allied("gov", "reb") is False
        e.form_alliance("gov", "reb")
        assert e.are_allied("gov", "reb") is True


# ---------------------------------------------------------------------------
# Tick — event processing
# ---------------------------------------------------------------------------

class TestTick:
    def test_tick_advances_time(self):
        e = _engine_2()
        e.tick(1.0)
        assert e._sim_time == 1.0
        e.tick(0.5)
        assert e._sim_time == 1.5

    def test_tick_no_events(self):
        e = _engine_2()
        e.tick(1.0)  # no crash
        e.tick(1.0, [])  # no crash

    def test_attack_degrades_relation(self):
        e = _engine_2()
        e.tick(1.0, [{"type": "attack", "attacker_faction": "gov",
                       "target_faction": "reb"}])
        dr = e.get_relation("gov", "reb")
        assert dr.relation != Relation.NEUTRAL  # should have degraded
        assert dr.trust < 0.5

    def test_repeated_attacks_escalate_to_war(self):
        e = _engine_2()
        for _ in range(10):
            e.tick(1.0, [{"type": "attack", "attacker_faction": "gov",
                           "target_faction": "reb"}])
        dr = e.get_relation("gov", "reb")
        assert dr.relation == Relation.WAR

    def test_collateral_civilian_heavy_penalty(self):
        e = _engine_3()
        initial_trust = e.get_relation("gov", "civ").trust
        e.tick(1.0, [{"type": "collateral", "attacker_faction": "gov",
                       "victim_faction": "civ"}])
        dr = e.get_relation("gov", "civ")
        assert dr.trust < initial_trust - 0.2  # heavy penalty for civilians

    def test_collateral_non_civilian(self):
        e = _engine_2()
        initial_trust = e.get_relation("gov", "reb").trust
        e.tick(1.0, [{"type": "collateral", "attacker_faction": "gov",
                       "victim_faction": "reb"}])
        dr = e.get_relation("gov", "reb")
        assert dr.trust < initial_trust

    def test_shared_enemy_improves_trust(self):
        e = _engine_3()
        initial_trust = e.get_relation("gov", "civ").trust
        e.tick(1.0, [{"type": "shared_enemy", "faction_a": "gov",
                       "faction_b": "civ", "enemy": "reb"}])
        assert e.get_relation("gov", "civ").trust > initial_trust

    def test_shared_enemy_caps_at_friendly(self):
        e = _engine_2()
        e.form_alliance("gov", "reb")
        e.tick(1.0, [{"type": "shared_enemy", "faction_a": "gov",
                       "faction_b": "reb", "enemy": "civ"}])
        # Should not go past FRIENDLY (alliance requires explicit action)
        # But it was already ALLIED, so shared_enemy would move it to FRIENDLY
        # Actually ALLIED=0, floor is FRIENDLY=1, so it would set to FRIENDLY
        # This is intentional — shared_enemy alone doesn't create alliances
        dr = e.get_relation("gov", "reb")
        assert dr.relation in (Relation.ALLIED, Relation.FRIENDLY)

    def test_ceasefire_broken_instant_war(self):
        e = _engine_2()
        e.tick(1.0, [{"type": "ceasefire_broken", "breaker": "reb",
                       "victim": "gov"}])
        dr = e.get_relation("gov", "reb")
        assert dr.relation == Relation.WAR
        assert dr.trust == 0.0
        assert dr.ceasefire_until is None

    def test_trade_improves_trust(self):
        e = _engine_2()
        initial_trust = e.get_relation("gov", "reb").trust
        e.tick(1.0, [{"type": "trade", "faction_a": "gov", "faction_b": "reb"}])
        dr = e.get_relation("gov", "reb")
        assert dr.trust > initial_trust
        assert dr.trade_active is True

    def test_trade_blocked_during_war(self):
        e = _engine_2()
        e.declare_war("gov", "reb")
        initial_trust = e.get_relation("gov", "reb").trust
        e.tick(1.0, [{"type": "trade", "faction_a": "gov", "faction_b": "reb"}])
        # Trade should not improve trust during war
        assert e.get_relation("gov", "reb").trust == initial_trust

    def test_multiple_events_in_one_tick(self):
        e = _engine_3()
        events = [
            {"type": "attack", "attacker_faction": "reb", "target_faction": "gov"},
            {"type": "collateral", "attacker_faction": "reb", "victim_faction": "civ"},
            {"type": "shared_enemy", "faction_a": "gov", "faction_b": "civ", "enemy": "reb"},
        ]
        e.tick(1.0, events)
        # reb attacked gov -> gov-reb degraded
        assert e.get_relation("gov", "reb").trust < 0.5
        # reb collateral on civ -> civ-reb degraded
        # shared enemy -> gov-civ improved
        assert len(e.get_relation("gov", "reb").history) >= 1

    def test_unknown_event_type_ignored(self):
        e = _engine_2()
        e.tick(1.0, [{"type": "unknown_thing", "data": 42}])  # no crash


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

class TestQueries:
    def test_get_enemies(self):
        e = _engine_3()
        e.declare_war("gov", "reb")
        enemies = e.get_enemies("gov")
        assert "reb" in enemies
        assert "civ" not in enemies

    def test_get_enemies_empty(self):
        e = _engine_2()
        assert e.get_enemies("gov") == []

    def test_get_allies(self):
        e = _engine_3()
        e.form_alliance("gov", "civ")
        allies = e.get_allies("gov")
        assert "civ" in allies
        assert "reb" not in allies

    def test_get_allies_empty(self):
        e = _engine_2()
        assert e.get_allies("gov") == []


# ---------------------------------------------------------------------------
# Export / visualization
# ---------------------------------------------------------------------------

class TestExport:
    def test_diplomatic_map_structure(self):
        e = _engine_2()
        m = e.get_diplomatic_map()
        assert "factions" in m
        assert "relations" in m
        assert "sim_time" in m
        assert "gov" in m["factions"]
        assert len(m["relations"]) == 1

    def test_diplomatic_map_faction_fields(self):
        e = _engine_2()
        m = e.get_diplomatic_map()
        f = m["factions"]["gov"]
        assert f["name"] == "Government"
        assert f["ideology"] == "government"
        assert f["strength"] == 0.8

    def test_diplomatic_map_relation_fields(self):
        e = _engine_2()
        e.set_relation("gov", "reb", Relation.HOSTILE)
        m = e.get_diplomatic_map()
        r = m["relations"][0]
        assert r["relation"] == "hostile"
        assert "trust" in r
        assert "trade_active" in r

    def test_to_three_js_structure(self):
        e = _engine_2()
        js = e.to_three_js()
        assert "factions" in js
        assert "relations" in js

    def test_to_three_js_faction_has_centroid(self):
        f = _gov()
        f.territory = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        e = DiplomacyEngine()
        e.add_faction(f)
        js = e.to_three_js()
        fc = js["factions"][0]
        assert fc["centroid"] is not None
        assert len(fc["centroid"]) == 2

    def test_to_three_js_relation_colors(self):
        e = _engine_2()
        e.declare_war("gov", "reb")
        js = e.to_three_js()
        rel = js["relations"][0]
        assert rel["color"] == "#ff2a6d"  # war = magenta

    def test_to_three_js_empty_territory(self):
        e = _engine_2()
        js = e.to_three_js()
        fc = [f for f in js["factions"] if f["id"] == "gov"][0]
        assert fc["centroid"] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

class TestPresets:
    def test_all_presets_exist(self):
        assert "three_way_war" in FACTION_PRESETS
        assert "proxy_conflict" in FACTION_PRESETS
        assert "insurgency" in FACTION_PRESETS
        assert "peacekeeping" in FACTION_PRESETS

    def test_load_three_way_war(self):
        e = load_preset("three_way_war")
        assert len(e.factions) == 3
        assert e.are_hostile("alpha", "bravo")
        assert e.are_hostile("alpha", "charlie")

    def test_load_proxy_conflict(self):
        e = load_preset("proxy_conflict")
        assert len(e.factions) == 4
        assert e.are_allied("east", "proxy_e")
        assert e.are_allied("west", "proxy_w")
        assert e.are_hostile("proxy_e", "proxy_w")

    def test_load_insurgency(self):
        e = load_preset("insurgency")
        assert len(e.factions) == 3
        assert e.are_hostile("gov", "reb")
        assert not e.are_hostile("gov", "civ")

    def test_load_peacekeeping(self):
        e = load_preset("peacekeeping")
        assert len(e.factions) == 4
        assert e.are_hostile("gov", "reb")
        assert not e.are_hostile("un", "reb")

    def test_load_invalid_preset(self):
        with pytest.raises(KeyError):
            load_preset("nonexistent")

    def test_preset_descriptions(self):
        for name, preset in FACTION_PRESETS.items():
            assert "description" in preset
            assert "factions" in preset

    def test_preset_factions_have_required_fields(self):
        for name, preset in FACTION_PRESETS.items():
            for f in preset["factions"]:
                assert f.faction_id
                assert f.name
                assert f.color
                assert f.ideology


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_centroid_empty(self):
        assert _centroid([]) == (0.0, 0.0)

    def test_centroid_single(self):
        assert _centroid([(5.0, 3.0)]) == (5.0, 3.0)

    def test_centroid_multiple(self):
        c = _centroid([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        assert abs(c[0] - 5.0) < 0.001
        assert abs(c[1] - 5.0) < 0.001

    def test_relation_colors(self):
        assert _relation_color(Relation.WAR) == "#ff2a6d"
        assert _relation_color(Relation.ALLIED) == "#05ffa1"
        assert _relation_color(Relation.NEUTRAL) == "#888888"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_trust_never_below_zero(self):
        e = _engine_2()
        for _ in range(50):
            e.tick(1.0, [{"type": "attack", "attacker_faction": "gov",
                           "target_faction": "reb"}])
        assert e.get_relation("gov", "reb").trust >= 0.0

    def test_trust_never_above_one(self):
        e = _engine_2()
        for _ in range(50):
            e.tick(1.0, [{"type": "trade", "faction_a": "gov", "faction_b": "reb"}])
        assert e.get_relation("gov", "reb").trust <= 1.0

    def test_self_relation_not_created(self):
        """Getting relation between a faction and itself shouldn't crash."""
        e = DiplomacyEngine()
        e.add_faction(_gov())
        dr = e.get_relation("gov", "gov")
        assert dr.relation == Relation.NEUTRAL

    def test_war_history_preserved(self):
        e = _engine_2()
        e.declare_war("gov", "reb")
        e.form_alliance("gov", "reb")
        dr = e.get_relation("gov", "reb")
        assert len(dr.history) >= 2

    def test_tick_with_none_events(self):
        e = _engine_2()
        e.tick(1.0, None)  # should not crash

    def test_many_factions(self):
        e = DiplomacyEngine()
        for i in range(20):
            e.add_faction(Faction(
                faction_id=f"f{i}", name=f"Faction {i}", color=f"#{i:06x}",
                ideology="mercenary", strength=0.5, wealth=0.5,
            ))
        # 20 factions -> 190 pairwise relations
        assert len(e.relations) == 190
        assert e.get_relation("f0", "f19").relation == Relation.NEUTRAL

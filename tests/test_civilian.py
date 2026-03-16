# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the civilian population and infrastructure simulation module."""

from __future__ import annotations

import math
import pytest

from tritium_lib.sim_engine.civilian import (
    Civilian,
    CivilianSimulator,
    CivilianState,
    CollateralDamage,
    Infrastructure,
    InfrastructureType,
    INFRASTRUCTURE_TEMPLATES,
    _FEAR_FLEE_THRESHOLD,
    _FEAR_SHELTER_THRESHOLD,
)
from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# CivilianState enum
# ---------------------------------------------------------------------------


class TestCivilianState:
    def test_values(self):
        assert CivilianState.NORMAL == 0
        assert CivilianState.SHELTERING == 1
        assert CivilianState.FLEEING == 2
        assert CivilianState.INJURED == 3
        assert CivilianState.DEAD == 4

    def test_all_states_exist(self):
        assert len(CivilianState) == 5

    def test_ordering(self):
        assert CivilianState.NORMAL < CivilianState.DEAD

    def test_name_access(self):
        assert CivilianState.FLEEING.name == "FLEEING"


# ---------------------------------------------------------------------------
# InfrastructureType enum
# ---------------------------------------------------------------------------


class TestInfrastructureType:
    def test_values(self):
        assert InfrastructureType.POWER_PLANT == 0
        assert InfrastructureType.TELECOM_TOWER == 7

    def test_all_types_exist(self):
        assert len(InfrastructureType) == 8

    def test_each_type_has_template(self):
        for itype in InfrastructureType:
            assert itype in INFRASTRUCTURE_TEMPLATES


# ---------------------------------------------------------------------------
# Civilian dataclass
# ---------------------------------------------------------------------------


class TestCivilian:
    def test_defaults(self):
        c = Civilian(civilian_id="c1", position=(10.0, 20.0))
        assert c.state == CivilianState.NORMAL
        assert c.speed == 3.0
        assert c.fear == 0.0
        assert c.health == 100.0
        assert c.home_position == (0.0, 0.0)
        assert c.work_position is None
        assert c.destination is None

    def test_custom_values(self):
        c = Civilian(
            civilian_id="c2",
            position=(5.0, 5.0),
            state=CivilianState.FLEEING,
            home_position=(0.0, 0.0),
            work_position=(100.0, 100.0),
            speed=5.0,
            fear=0.8,
            health=60.0,
        )
        assert c.state == CivilianState.FLEEING
        assert c.speed == 5.0
        assert c.fear == 0.8
        assert c.health == 60.0

    def test_mutable_state(self):
        c = Civilian(civilian_id="c3", position=(0.0, 0.0))
        c.state = CivilianState.DEAD
        c.health = 0.0
        assert c.state == CivilianState.DEAD


# ---------------------------------------------------------------------------
# Infrastructure dataclass
# ---------------------------------------------------------------------------


class TestInfrastructure:
    def test_defaults(self):
        i = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.HOSPITAL,
            position=(50.0, 50.0),
            radius=300.0,
        )
        assert i.health == 100.0
        assert i.max_health == 100.0
        assert i.is_operational is True
        assert i.serves_population == 0
        assert i.repair_rate == 0.1

    def test_custom_health(self):
        i = Infrastructure(
            infra_id="i2",
            infra_type=InfrastructureType.POWER_PLANT,
            position=(0.0, 0.0),
            radius=500.0,
            health=200.0,
            max_health=200.0,
        )
        assert i.health == 200.0


# ---------------------------------------------------------------------------
# CollateralDamage dataclass
# ---------------------------------------------------------------------------


class TestCollateralDamage:
    def test_defaults(self):
        cd = CollateralDamage(event_id="e1", position=(10.0, 10.0), timestamp=1.0)
        assert cd.civilian_casualties == 0
        assert cd.infrastructure_damage == []
        assert cd.cause == "unknown"
        assert cd.severity == 0.0
        assert cd.hearts_minds_impact == 0.0

    def test_custom(self):
        cd = CollateralDamage(
            event_id="e2",
            position=(0.0, 0.0),
            timestamp=5.0,
            civilian_casualties=3,
            infrastructure_damage=["i1", "i2"],
            cause="airstrike",
            severity=0.8,
            hearts_minds_impact=-0.3,
        )
        assert cd.cause == "airstrike"
        assert len(cd.infrastructure_damage) == 2


# ---------------------------------------------------------------------------
# INFRASTRUCTURE_TEMPLATES
# ---------------------------------------------------------------------------


class TestInfrastructureTemplates:
    def test_all_types_have_keys(self):
        for itype, tpl in INFRASTRUCTURE_TEMPLATES.items():
            assert "health" in tpl
            assert "radius" in tpl
            assert "population_capacity" in tpl
            assert "repair_rate" in tpl

    def test_positive_values(self):
        for tpl in INFRASTRUCTURE_TEMPLATES.values():
            assert tpl["health"] > 0
            assert tpl["radius"] > 0
            assert tpl["repair_rate"] > 0

    def test_power_plant_large_radius(self):
        tpl = INFRASTRUCTURE_TEMPLATES[InfrastructureType.POWER_PLANT]
        assert tpl["radius"] >= 400

    def test_telecom_largest_radius(self):
        tpl = INFRASTRUCTURE_TEMPLATES[InfrastructureType.TELECOM_TOWER]
        assert tpl["radius"] >= 800


# ---------------------------------------------------------------------------
# CivilianSimulator — spawning
# ---------------------------------------------------------------------------


class TestSpawning:
    def test_spawn_basic(self):
        sim = CivilianSimulator()
        sim.spawn_population((100.0, 100.0), 50, 200.0)
        assert len(sim.civilians) == 50

    def test_spawn_with_infrastructure(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 10, 100.0, with_infrastructure=True)
        assert len(sim.infrastructure) > 0

    def test_spawn_without_infrastructure(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 10, 100.0, with_infrastructure=False)
        assert len(sim.infrastructure) == 0

    def test_civilians_within_radius(self):
        sim = CivilianSimulator()
        center = (500.0, 500.0)
        sim.spawn_population(center, 100, 200.0, with_infrastructure=False)
        for civ in sim.civilians:
            d = distance(civ.position, center)
            assert d <= 200.0 + 1.0  # small float tolerance

    def test_civilians_start_normal(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 20, 50.0, with_infrastructure=False)
        for civ in sim.civilians:
            assert civ.state == CivilianState.NORMAL
            assert civ.fear == 0.0
            assert civ.health == 100.0

    def test_multiple_spawns_accumulate(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 10, 50.0, with_infrastructure=False)
        sim.spawn_population((500.0, 500.0), 15, 50.0, with_infrastructure=False)
        assert len(sim.civilians) == 25

    def test_infrastructure_types_spawned(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 10, 200.0)
        types_spawned = {i.infra_type for i in sim.infrastructure.values()}
        assert InfrastructureType.HOSPITAL in types_spawned
        assert InfrastructureType.POWER_PLANT in types_spawned


# ---------------------------------------------------------------------------
# CivilianSimulator — tick basics
# ---------------------------------------------------------------------------


class TestTickBasics:
    def test_tick_no_threats(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 20, 100.0, with_infrastructure=False)
        result = sim.tick(1.0)
        assert result["casualties"] == 0
        assert result["injured"] == 0
        assert "sentiment" in result
        assert "fear_avg" in result

    def test_tick_returns_dict(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 5, 50.0, with_infrastructure=False)
        result = sim.tick(0.5)
        assert isinstance(result, dict)

    def test_empty_sim_tick(self):
        sim = CivilianSimulator()
        result = sim.tick(1.0)
        assert result["casualties"] == 0


# ---------------------------------------------------------------------------
# CivilianSimulator — explosions
# ---------------------------------------------------------------------------


class TestExplosions:
    def test_explosion_kills_nearby(self):
        sim = CivilianSimulator()
        # place a civilian right at the explosion
        civ = Civilian(civilian_id="c1", position=(0.0, 0.0))
        sim.civilians.append(civ)
        result = sim.tick(1.0, explosions=[((0.0, 0.0), 20.0)])
        assert civ.state == CivilianState.DEAD
        assert result["casualties"] >= 1

    def test_explosion_injures_at_range(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(30.0, 0.0))
        sim.civilians.append(civ)
        sim.tick(1.0, explosions=[((0.0, 0.0), 20.0)])
        assert civ.health < 100.0

    def test_explosion_creates_collateral_event(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(5.0, 0.0))
        sim.civilians.append(civ)
        sim.tick(1.0, explosions=[((0.0, 0.0), 20.0)])
        assert len(sim.collateral_events) == 1
        assert sim.collateral_events[0].cause == "explosion"

    def test_explosion_damages_infrastructure(self):
        sim = CivilianSimulator()
        infra = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.HOSPITAL,
            position=(10.0, 0.0),
            radius=300.0,
            health=100.0,
            max_health=100.0,
        )
        sim.infrastructure["i1"] = infra
        sim.tick(1.0, explosions=[((0.0, 0.0), 20.0)])
        assert infra.health < 100.0

    def test_explosion_reduces_sentiment(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(5.0, 0.0))
        sim.civilians.append(civ)
        initial = sim.population_sentiment
        sim.tick(1.0, explosions=[((0.0, 0.0), 20.0)])
        assert sim.population_sentiment < initial

    def test_far_civilian_not_killed(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(500.0, 500.0))
        sim.civilians.append(civ)
        sim.tick(1.0, explosions=[((0.0, 0.0), 20.0)])
        assert civ.state != CivilianState.DEAD
        assert civ.health == 100.0

    def test_multiple_explosions(self):
        sim = CivilianSimulator()
        for i in range(5):
            sim.civilians.append(Civilian(civilian_id=f"c{i}", position=(float(i * 100), 0.0)))
        sim.tick(1.0, explosions=[((0.0, 0.0), 20.0), ((100.0, 0.0), 20.0)])
        dead = sum(1 for c in sim.civilians if c.state == CivilianState.DEAD)
        assert dead >= 2

    def test_dead_civilian_not_affected_again(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(0.0, 0.0), state=CivilianState.DEAD, health=0.0)
        sim.civilians.append(civ)
        sim.tick(1.0, explosions=[((0.0, 0.0), 20.0)])
        assert civ.health == 0.0


# ---------------------------------------------------------------------------
# CivilianSimulator — fear and fleeing
# ---------------------------------------------------------------------------


class TestFearAndFleeing:
    def test_threat_increases_fear(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(50.0, 0.0))
        sim.civilians.append(civ)
        sim.tick(1.0, threats=[((0.0, 0.0), 100.0)])
        assert civ.fear > 0.0

    def test_high_fear_causes_fleeing(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(50.0, 0.0), fear=0.9)
        sim.civilians.append(civ)
        sim.tick(1.0, threats=[((0.0, 0.0), 100.0)])
        assert civ.state == CivilianState.FLEEING

    def test_moderate_fear_causes_sheltering(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(50.0, 0.0), fear=_FEAR_SHELTER_THRESHOLD + 0.05)
        sim.civilians.append(civ)
        sim.tick(0.1)  # no threats, but fear already high enough
        assert civ.state == CivilianState.SHELTERING

    def test_fear_decays_without_threats(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(0.0, 0.0), fear=0.5)
        sim.civilians.append(civ)
        sim.tick(1.0)
        assert civ.fear < 0.5

    def test_fleeing_moves_away_from_threat(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(50.0, 0.0), fear=0.9)
        sim.civilians.append(civ)
        old_x = civ.position[0]
        sim.tick(1.0, threats=[((0.0, 0.0), 100.0)])
        # should move away (increasing x)
        assert civ.position[0] > old_x

    def test_fear_clamped_to_1(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(5.0, 0.0), fear=0.95)
        sim.civilians.append(civ)
        sim.tick(1.0, threats=[((0.0, 0.0), 200.0)])
        assert civ.fear <= 1.0

    def test_fear_does_not_go_negative(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(0.0, 0.0), fear=0.01)
        sim.civilians.append(civ)
        sim.tick(10.0)
        assert civ.fear >= 0.0


# ---------------------------------------------------------------------------
# CivilianSimulator — humanitarian aid
# ---------------------------------------------------------------------------


class TestAid:
    def test_aid_improves_sentiment(self):
        sim = CivilianSimulator()
        sim.population_sentiment = 0.3
        sim.spawn_population((0.0, 0.0), 20, 50.0, with_infrastructure=False)
        improvement = sim.provide_aid((0.0, 0.0), 200.0, 5.0)
        assert improvement > 0
        assert sim.population_sentiment > 0.3

    def test_aid_reduces_fear(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(0.0, 0.0), fear=0.8)
        sim.civilians.append(civ)
        sim.provide_aid((0.0, 0.0), 50.0, 5.0)
        assert civ.fear < 0.8

    def test_aid_heals_injured(self):
        sim = CivilianSimulator()
        civ = Civilian(
            civilian_id="c1",
            position=(0.0, 0.0),
            state=CivilianState.INJURED,
            health=30.0,
        )
        sim.civilians.append(civ)
        sim.provide_aid((0.0, 0.0), 50.0, 20.0)
        assert civ.health > 30.0

    def test_aid_no_effect_on_dead(self):
        sim = CivilianSimulator()
        civ = Civilian(
            civilian_id="c1",
            position=(0.0, 0.0),
            state=CivilianState.DEAD,
            health=0.0,
        )
        sim.civilians.append(civ)
        improvement = sim.provide_aid((0.0, 0.0), 50.0, 10.0)
        assert improvement == 0.0
        assert civ.health == 0.0

    def test_aid_out_of_range(self):
        sim = CivilianSimulator()
        civ = Civilian(civilian_id="c1", position=(1000.0, 1000.0))
        sim.civilians.append(civ)
        improvement = sim.provide_aid((0.0, 0.0), 50.0, 10.0)
        assert improvement == 0.0

    def test_aid_sentiment_capped_at_1(self):
        sim = CivilianSimulator()
        sim.population_sentiment = 0.99
        civ = Civilian(civilian_id="c1", position=(0.0, 0.0))
        sim.civilians.append(civ)
        sim.provide_aid((0.0, 0.0), 50.0, 100.0)
        assert sim.population_sentiment <= 1.0


# ---------------------------------------------------------------------------
# CivilianSimulator — infrastructure repair
# ---------------------------------------------------------------------------


class TestRepair:
    def test_repair_restores_health(self):
        sim = CivilianSimulator()
        infra = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.BRIDGE,
            position=(0.0, 0.0),
            radius=30.0,
            health=50.0,
            max_health=100.0,
            repair_rate=1.0,
        )
        sim.infrastructure["i1"] = infra
        restored = sim.repair_infrastructure("i1", engineers=2, dt=5.0)
        assert restored > 0
        assert infra.health > 50.0

    def test_repair_does_not_exceed_max(self):
        sim = CivilianSimulator()
        infra = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.ROAD,
            position=(0.0, 0.0),
            radius=50.0,
            health=99.0,
            max_health=100.0,
            repair_rate=10.0,
        )
        sim.infrastructure["i1"] = infra
        sim.repair_infrastructure("i1", engineers=5, dt=10.0)
        assert infra.health == 100.0

    def test_repair_reactivates_infrastructure(self):
        sim = CivilianSimulator()
        infra = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.HOSPITAL,
            position=(0.0, 0.0),
            radius=300.0,
            health=10.0,
            max_health=100.0,
            is_operational=False,
            repair_rate=5.0,
        )
        sim.infrastructure["i1"] = infra
        sim.repair_infrastructure("i1", engineers=3, dt=5.0)
        assert infra.health >= 20.0
        assert infra.is_operational is True

    def test_repair_unknown_id(self):
        sim = CivilianSimulator()
        result = sim.repair_infrastructure("nonexistent", engineers=1, dt=1.0)
        assert result == 0.0

    def test_repair_already_full(self):
        sim = CivilianSimulator()
        infra = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.MARKET,
            position=(0.0, 0.0),
            radius=150.0,
            health=100.0,
            max_health=100.0,
        )
        sim.infrastructure["i1"] = infra
        result = sim.repair_infrastructure("i1", engineers=1, dt=1.0)
        assert result == 0.0

    def test_repair_improves_sentiment(self):
        sim = CivilianSimulator()
        initial = sim.population_sentiment
        infra = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.POWER_PLANT,
            position=(0.0, 0.0),
            radius=500.0,
            health=10.0,
            max_health=200.0,
            repair_rate=5.0,
        )
        sim.infrastructure["i1"] = infra
        sim.repair_infrastructure("i1", engineers=5, dt=10.0)
        assert sim.population_sentiment >= initial


# ---------------------------------------------------------------------------
# CivilianSimulator — population report
# ---------------------------------------------------------------------------


class TestPopulationReport:
    def test_report_keys(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 10, 50.0, with_infrastructure=False)
        report = sim.get_population_report()
        for key in ["total", "alive", "injured", "dead", "sheltering", "fleeing",
                     "normal", "sentiment", "collateral_events",
                     "infrastructure_operational", "infrastructure_total"]:
            assert key in report

    def test_report_counts(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 30, 50.0, with_infrastructure=False)
        report = sim.get_population_report()
        assert report["total"] == 30
        assert report["alive"] == 30
        assert report["dead"] == 0

    def test_report_after_casualties(self):
        sim = CivilianSimulator()
        sim.civilians.append(Civilian(civilian_id="c1", position=(0.0, 0.0)))
        sim.civilians.append(Civilian(civilian_id="c2", position=(500.0, 500.0)))
        sim.tick(1.0, explosions=[((0.0, 0.0), 20.0)])
        report = sim.get_population_report()
        assert report["dead"] >= 1
        assert report["alive"] + report["dead"] == report["total"]

    def test_report_infrastructure_counts(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 5, 100.0, with_infrastructure=True)
        report = sim.get_population_report()
        assert report["infrastructure_total"] > 0
        assert report["infrastructure_operational"] <= report["infrastructure_total"]


# ---------------------------------------------------------------------------
# CivilianSimulator — to_three_js
# ---------------------------------------------------------------------------


class TestToThreeJs:
    def test_output_structure(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 5, 50.0)
        out = sim.to_three_js()
        assert "civilians" in out
        assert "infrastructure" in out
        assert "sentiment" in out
        assert "sentiment_color" in out
        assert "casualties" in out

    def test_civilian_fields(self):
        sim = CivilianSimulator()
        sim.civilians.append(Civilian(civilian_id="c1", position=(10.0, 20.0), fear=0.5))
        out = sim.to_three_js()
        civ = out["civilians"][0]
        assert civ["id"] == "c1"
        assert civ["x"] == 10.0
        assert civ["y"] == 20.0
        assert civ["state"] == "normal"
        assert civ["fear"] == 0.5

    def test_infrastructure_fields(self):
        sim = CivilianSimulator()
        infra = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.HOSPITAL,
            position=(100.0, 100.0),
            radius=300.0,
            health=60.0,
            max_health=100.0,
        )
        sim.infrastructure["i1"] = infra
        out = sim.to_three_js()
        i = out["infrastructure"][0]
        assert i["id"] == "i1"
        assert i["type"] == "hospital"
        assert i["health_pct"] == 0.6
        assert i["operational"] is True
        assert i["radius"] == 300.0

    def test_sentiment_color_hostile(self):
        sim = CivilianSimulator()
        sim.population_sentiment = 0.0
        out = sim.to_three_js()
        assert out["sentiment_color"] == "#ff0000"

    def test_sentiment_color_neutral(self):
        sim = CivilianSimulator()
        sim.population_sentiment = 0.5
        out = sim.to_three_js()
        assert out["sentiment_color"] == "#ffff00"

    def test_sentiment_color_supportive(self):
        sim = CivilianSimulator()
        sim.population_sentiment = 1.0
        out = sim.to_three_js()
        assert out["sentiment_color"] == "#00ff00"

    def test_casualties_dict(self):
        sim = CivilianSimulator()
        sim.spawn_population((0.0, 0.0), 10, 50.0, with_infrastructure=False)
        out = sim.to_three_js()
        cas = out["casualties"]
        assert "alive" in cas
        assert "injured" in cas
        assert "dead" in cas


# ---------------------------------------------------------------------------
# CivilianSimulator — infrastructure cascades
# ---------------------------------------------------------------------------


class TestInfrastructureCascades:
    def test_destroyed_infra_not_operational(self):
        sim = CivilianSimulator()
        infra = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.POWER_PLANT,
            position=(0.0, 0.0),
            radius=500.0,
            health=0.0,
            max_health=200.0,
        )
        sim.infrastructure["i1"] = infra
        sim.tick(1.0)
        assert infra.is_operational is False

    def test_low_health_infra_not_operational(self):
        sim = CivilianSimulator()
        infra = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.WATER_TREATMENT,
            position=(0.0, 0.0),
            radius=400.0,
            health=10.0,
            max_health=150.0,
        )
        sim.infrastructure["i1"] = infra
        sim.tick(1.0)
        # 10/150 < 0.2 threshold
        assert infra.is_operational is False

    def test_healthy_infra_stays_operational(self):
        sim = CivilianSimulator()
        infra = Infrastructure(
            infra_id="i1",
            infra_type=InfrastructureType.SCHOOL,
            position=(0.0, 0.0),
            radius=200.0,
            health=80.0,
            max_health=80.0,
        )
        sim.infrastructure["i1"] = infra
        sim.tick(1.0)
        assert infra.is_operational is True


# ---------------------------------------------------------------------------
# CivilianSimulator — sentiment
# ---------------------------------------------------------------------------


class TestSentiment:
    def test_initial_sentiment(self):
        sim = CivilianSimulator()
        assert sim.population_sentiment == 0.5

    def test_sentiment_never_below_zero(self):
        sim = CivilianSimulator()
        sim.population_sentiment = 0.01
        # massive explosion with many civilians
        for i in range(50):
            sim.civilians.append(Civilian(civilian_id=f"c{i}", position=(float(i), 0.0)))
        sim.tick(1.0, explosions=[((25.0, 0.0), 50.0)])
        assert sim.population_sentiment >= 0.0

    def test_sentiment_never_above_one(self):
        sim = CivilianSimulator()
        sim.population_sentiment = 0.99
        civ = Civilian(civilian_id="c1", position=(0.0, 0.0))
        sim.civilians.append(civ)
        sim.provide_aid((0.0, 0.0), 100.0, 1000.0)
        assert sim.population_sentiment <= 1.0

    def test_sentiment_drifts_to_neutral(self):
        sim = CivilianSimulator()
        sim.population_sentiment = 0.2
        sim.civilians.append(Civilian(civilian_id="c1", position=(1000.0, 1000.0)))
        for _ in range(100):
            sim.tick(1.0)
        assert sim.population_sentiment > 0.2


# ---------------------------------------------------------------------------
# Integration — full scenario
# ---------------------------------------------------------------------------


class TestIntegrationScenario:
    def test_full_cycle(self):
        """Spawn, explode, aid, repair, report."""
        sim = CivilianSimulator()
        sim.spawn_population((200.0, 200.0), 100, 300.0)
        assert len(sim.civilians) == 100

        # peaceful tick
        r1 = sim.tick(1.0)
        assert r1["casualties"] == 0

        # explosion
        r2 = sim.tick(1.0, explosions=[((200.0, 200.0), 30.0)])
        assert r2["casualties"] >= 0  # may or may not kill depending on random positions

        # provide aid
        improvement = sim.provide_aid((200.0, 200.0), 500.0, 10.0)
        assert improvement >= 0

        # repair any damaged infrastructure
        for iid, infra in sim.infrastructure.items():
            if infra.health < infra.max_health:
                sim.repair_infrastructure(iid, engineers=3, dt=10.0)

        # final report
        report = sim.get_population_report()
        assert report["alive"] + report["dead"] == report["total"]

        # three.js export
        out = sim.to_three_js()
        assert len(out["civilians"]) == 100
        assert len(out["infrastructure"]) > 0

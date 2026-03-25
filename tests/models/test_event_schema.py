# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.models.event_schema — typed event schemas."""

from tritium_lib.models.event_schema import (
    ALL_EVENT_TYPES,
    AmmoDepletedEvent,
    AmmoLowEvent,
    BleUpdateEvent,
    DetectionEvent,
    DeviceHeartbeatEvent,
    DossierCreatedEvent,
    EliminationStreakEvent,
    EscalationChangeEvent,
    EventDomain,
    FederationSiteAddedEvent,
    FormationCreatedEvent,
    GameOverEvent,
    GameStateChangeEvent,
    HazardExpiredEvent,
    HazardSpawnedEvent,
    MeshMessageEvent,
    MissionProgressEvent,
    ModeChangeEvent,
    NpcAllianceChangeEvent,
    NpcThoughtClearEvent,
    NpcThoughtEvent,
    ProjectileFiredEvent,
    ProjectileHitEvent,
    ScenarioGeneratedEvent,
    SensorClearedEvent,
    SensorTriggeredEvent,
    SimTelemetryBatchEvent,
    SimTelemetryEvent,
    TakContactEvent,
    TargetEliminatedEvent,
    TritiumEvent,
    UnitDispatchedEvent,
    UnitSignalEvent,
    WaveCompleteEvent,
    WaveStartEvent,
    WeaponJamEvent,
    WiFiUpdateEvent,
    ZoneViolationEvent,
    get_event_schema,
    list_event_types,
    validate_event_type,
)


class TestEventDomain:
    def test_all_domains_exist(self):
        assert EventDomain.SIMULATION == "simulation"
        assert EventDomain.COMBAT == "combat"
        assert EventDomain.GAME == "game"
        assert EventDomain.NPC == "npc"
        assert EventDomain.FLEET == "fleet"
        assert EventDomain.MESH == "mesh"
        assert EventDomain.EDGE == "edge"
        assert EventDomain.TAK == "tak"
        assert EventDomain.SENSOR == "sensor"
        assert EventDomain.TARGET == "target"
        assert EventDomain.DOSSIER == "dossier"
        assert EventDomain.FEDERATION == "federation"
        assert EventDomain.AMY == "amy"
        assert EventDomain.AUDIO == "audio"
        assert EventDomain.MISSION == "mission"
        assert EventDomain.HAZARD == "hazard"
        assert EventDomain.UNIT == "unit"


class TestTritiumEvent:
    def test_base_event_creation(self):
        e = TritiumEvent(event_type="test", domain=EventDomain.SIMULATION)
        assert e.event_type == "test"
        assert e.domain == EventDomain.SIMULATION
        assert e.description == ""


class TestSimEvents:
    def test_sim_telemetry(self):
        e = SimTelemetryEvent()
        assert e.event_type == "sim_telemetry"
        assert e.domain == EventDomain.SIMULATION
        assert e.position_x == 0.0
        assert e.alliance == "unknown"

    def test_sim_telemetry_batch(self):
        e = SimTelemetryBatchEvent()
        assert e.event_type == "sim_telemetry_batch"


class TestGameEvents:
    def test_game_state_change(self):
        e = GameStateChangeEvent(state="playing", wave=3, score=500)
        assert e.state == "playing"
        assert e.wave == 3
        assert e.score == 500

    def test_wave_start(self):
        e = WaveStartEvent(wave=2, hostile_count=10)
        assert e.hostile_count == 10

    def test_wave_complete(self):
        e = WaveCompleteEvent(wave=5)
        assert e.wave == 5

    def test_game_over(self):
        e = GameOverEvent(victory=True, score=5000, waves_completed=10)
        assert e.victory is True
        assert e.waves_completed == 10


class TestCombatEvents:
    def test_projectile_fired(self):
        e = ProjectileFiredEvent(source_id="rover_1", target_x=10.0, target_y=20.0)
        assert e.source_id == "rover_1"

    def test_projectile_hit(self):
        e = ProjectileHitEvent(target_id="hostile_1", damage=25.0)
        assert e.damage == 25.0

    def test_target_eliminated(self):
        e = TargetEliminatedEvent(target_id="h1", eliminated_by="r1")
        assert e.target_id == "h1"

    def test_elimination_streak(self):
        e = EliminationStreakEvent(unit_id="r1", streak=3)
        assert e.streak == 3

    def test_weapon_jam(self):
        e = WeaponJamEvent(unit_id="r1")
        assert e.event_type == "weapon_jam"

    def test_ammo_depleted(self):
        e = AmmoDepletedEvent(unit_id="r1")
        assert e.event_type == "ammo_depleted"

    def test_ammo_low(self):
        e = AmmoLowEvent(unit_id="r1", remaining=5)
        assert e.remaining == 5


class TestNpcEvents:
    def test_npc_thought(self):
        e = NpcThoughtEvent(npc_id="civ1", thought="What a day")
        assert e.thought == "What a day"

    def test_npc_thought_clear(self):
        e = NpcThoughtClearEvent(npc_id="civ1")
        assert e.event_type == "npc_thought_clear"

    def test_npc_alliance_change(self):
        e = NpcAllianceChangeEvent(npc_id="civ1", old_alliance="neutral", new_alliance="hostile")
        assert e.new_alliance == "hostile"


class TestRegistryFunctions:
    def test_validate_event_type_known(self):
        assert validate_event_type("sim_telemetry") is True
        assert validate_event_type("game_over") is True
        assert validate_event_type("projectile_hit") is True

    def test_validate_event_type_unknown(self):
        assert validate_event_type("nonexistent_event") is False

    def test_get_event_schema_known(self):
        schema = get_event_schema("sim_telemetry")
        assert schema is SimTelemetryEvent

    def test_get_event_schema_unknown(self):
        assert get_event_schema("no_such_event") is None

    def test_list_event_types(self):
        types = list_event_types()
        assert len(types) > 30
        # Each entry has required keys
        for entry in types:
            assert "event_type" in entry
            assert "domain" in entry
            assert "description" in entry

    def test_all_event_types_registry_complete(self):
        """Every class in ALL_EVENT_TYPES can be instantiated."""
        for name, cls in ALL_EVENT_TYPES.items():
            instance = cls()
            assert instance.event_type == name

    def test_all_event_types_are_tritium_events(self):
        for name, cls in ALL_EVENT_TYPES.items():
            instance = cls()
            assert isinstance(instance, TritiumEvent)

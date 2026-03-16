# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the standalone sim_engine game server.

Uses httpx + FastAPI TestClient to verify all endpoints, game state
construction, and frame output.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tritium_lib.sim_engine.demos.game_server import (
    app,
    build_full_game,
    game_tick,
    GameState,
    _count_active_modules,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Fresh TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def game_state():
    """Build a full game state for testing."""
    return build_full_game("urban_combat")


# ---------------------------------------------------------------------------
# Server endpoint tests
# ---------------------------------------------------------------------------


class TestServerEndpoints:
    """Test the REST and WebSocket endpoints."""

    def test_index_serves_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Tritium Sim Engine" in resp.text
        assert "<canvas" in resp.text

    def test_status_before_start(self, client: TestClient) -> None:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False

    def test_presets_returns_lists(self, client: TestClient) -> None:
        resp = client.get("/api/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "urban_combat" in data["world_presets"]
        assert "skirmish" in data["scenario_presets"]
        assert "tutorial" in data["campaign_presets"]
        assert len(data["vehicle_templates"]) > 0
        assert len(data["aircraft_templates"]) > 0
        assert data["weapon_count"] > 30

    def test_start_creates_game(self, client: TestClient) -> None:
        resp = client.post("/api/start", json={"preset": "urban_combat"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert data["preset"] == "urban_combat"
        assert data["modules"] >= 10

    def test_status_after_start(self, client: TestClient) -> None:
        client.post("/api/start", json={"preset": "urban_combat"})
        resp = client.get("/api/status")
        data = resp.json()
        assert data["running"] is True
        assert data["preset"] == "urban_combat"
        assert "stats" in data
        assert "factions" in data

    def test_pause_toggles(self, client: TestClient) -> None:
        client.post("/api/start", json={"preset": "urban_combat"})
        resp = client.post("/api/pause")
        assert resp.json()["paused"] is True
        resp = client.post("/api/pause")
        assert resp.json()["paused"] is False

    def test_stats_returns_leaderboard(self, client: TestClient) -> None:
        client.post("/api/start", json={"preset": "urban_combat"})
        resp = client.get("/api/stats")
        data = resp.json()
        assert "leaderboard" in data
        assert "team_scores" in data

    def test_aar_returns_report(self, client: TestClient) -> None:
        client.post("/api/start", json={"preset": "urban_combat"})
        resp = client.get("/api/aar")
        data = resp.json()
        # AAR should have structure from ScoringEngine.generate_aar
        assert isinstance(data, dict)

    def test_command_move(self, client: TestClient) -> None:
        client.post("/api/start", json={"preset": "urban_combat"})
        # Find a valid unit ID from the game state
        from tritium_lib.sim_engine.demos.game_server import _game
        unit_id = next(iter(_game.world.units.keys()))
        resp = client.post("/api/command", json={
            "type": "move", "unit_id": unit_id, "target": [150, 150]
        })
        data = resp.json()
        assert data["status"] == "moved"

    def test_command_unknown(self, client: TestClient) -> None:
        client.post("/api/start", json={"preset": "urban_combat"})
        resp = client.post("/api/command", json={"type": "unknown"})
        assert resp.json()["status"] == "unknown_command"

    def test_pause_without_game(self, client: TestClient) -> None:
        import tritium_lib.sim_engine.demos.game_server as gs_mod
        gs_mod._game = GameState()  # Reset to fresh state
        resp = client.post("/api/pause")
        assert "error" in resp.json()


# ---------------------------------------------------------------------------
# Game state construction tests
# ---------------------------------------------------------------------------


class TestGameConstruction:
    """Test that build_full_game creates everything correctly."""

    def test_world_created(self, game_state: GameState) -> None:
        assert game_state.world is not None
        assert game_state.world.config.map_size == (500.0, 500.0)

    def test_units_spawned(self, game_state: GameState) -> None:
        world = game_state.world
        assert len(world.units) > 0
        friendly = [u for u in world.units.values() if u.alliance.value == "friendly"]
        hostile = [u for u in world.units.values() if u.alliance.value == "hostile"]
        assert len(friendly) >= 6  # 4 infantry + 1 sniper + 1 medic
        assert len(hostile) >= 8   # 6 infantry + 2 heavy

    def test_vehicles_spawned(self, game_state: GameState) -> None:
        world = game_state.world
        assert len(world.vehicles) >= 3  # humvee + technical + quadcopter

    def test_drone_controller_exists(self, game_state: GameState) -> None:
        assert len(game_state.world.drone_controllers) >= 1

    def test_structures_exist(self, game_state: GameState) -> None:
        assert game_state.world.destruction is not None
        assert len(game_state.world.destruction.structures) >= 4

    def test_crowd_exists(self, game_state: GameState) -> None:
        assert game_state.world.crowd is not None
        assert len(game_state.world.crowd.members) >= 40

    def test_scoring_initialized(self, game_state: GameState) -> None:
        assert game_state.scoring is not None
        assert len(game_state.scoring.unit_scores) > 0

    def test_detection_engine(self, game_state: GameState) -> None:
        assert game_state.detection is not None
        assert len(game_state.detection.sensors) > 0
        assert len(game_state.detection.signatures) > 0

    def test_comms_simulator(self, game_state: GameState) -> None:
        assert game_state.comms is not None
        assert len(game_state.comms.radios) > 0
        assert len(game_state.comms.channels) >= 2

    def test_medical_engine(self, game_state: GameState) -> None:
        assert game_state.medical is not None

    def test_logistics_engine(self, game_state: GameState) -> None:
        assert game_state.logistics is not None
        assert len(game_state.logistics.caches) >= 1

    def test_naval_engine(self, game_state: GameState) -> None:
        assert game_state.naval is not None
        assert len(game_state.naval.ships) >= 1

    def test_air_combat_engine(self, game_state: GameState) -> None:
        assert game_state.air_combat is not None
        assert len(game_state.air_combat.anti_air) >= 1

    def test_engineering_engine(self, game_state: GameState) -> None:
        assert game_state.engineering is not None
        assert len(game_state.engineering.fortifications) >= 2
        assert len(game_state.engineering.minefields) >= 8

    def test_asymmetric_engine(self, game_state: GameState) -> None:
        assert game_state.asymmetric is not None
        assert len(game_state.asymmetric.traps) >= 1

    def test_civilian_simulator(self, game_state: GameState) -> None:
        assert game_state.civilians is not None
        assert len(game_state.civilians.civilians) >= 40

    def test_intel_engine(self, game_state: GameState) -> None:
        assert game_state.intel is not None

    def test_diplomacy_engine(self, game_state: GameState) -> None:
        assert game_state.diplomacy is not None
        assert len(game_state.diplomacy.factions) == 3
        assert game_state.diplomacy.are_hostile("gov", "reb")

    def test_campaign_loaded(self, game_state: GameState) -> None:
        assert game_state.campaign is not None
        assert game_state.campaign.name == "Basic Training"

    def test_active_modules_count(self, game_state: GameState) -> None:
        count = _count_active_modules(game_state)
        assert count >= 14


# ---------------------------------------------------------------------------
# Game tick tests
# ---------------------------------------------------------------------------


class TestGameTick:
    """Test the tick function produces valid frames."""

    def test_tick_returns_frame(self, game_state: GameState) -> None:
        game_state.running = True
        frame = game_tick(game_state, dt=0.1)
        assert isinstance(frame, dict)
        assert "tick" in frame
        assert frame["tick"] == 1

    def test_frame_has_units(self, game_state: GameState) -> None:
        frame = game_tick(game_state, dt=0.1)
        assert "units" in frame
        assert len(frame["units"]) > 0

    def test_frame_has_vehicles(self, game_state: GameState) -> None:
        frame = game_tick(game_state, dt=0.1)
        assert "vehicles" in frame
        assert len(frame["vehicles"]) > 0

    def test_frame_has_detection(self, game_state: GameState) -> None:
        frame = game_tick(game_state, dt=0.1)
        assert "detection" in frame
        assert "sensors" in frame["detection"]

    def test_frame_has_comms(self, game_state: GameState) -> None:
        frame = game_tick(game_state, dt=0.1)
        assert "comms" in frame
        assert "radios" in frame["comms"]

    def test_frame_has_medical(self, game_state: GameState) -> None:
        frame = game_tick(game_state, dt=0.1)
        assert "medical" in frame

    def test_frame_has_logistics(self, game_state: GameState) -> None:
        frame = game_tick(game_state, dt=0.1)
        assert "logistics" in frame
        assert "caches" in frame["logistics"]

    def test_frame_has_naval(self, game_state: GameState) -> None:
        frame = game_tick(game_state, dt=0.1)
        assert "naval" in frame
        assert "ships" in frame["naval"]

    def test_frame_has_stats(self, game_state: GameState) -> None:
        frame = game_tick(game_state, dt=0.1)
        assert "stats" in frame
        stats = frame["stats"]
        assert "alive_friendly" in stats
        assert "alive_hostile" in stats
        assert stats["total_units"] > 0

    def test_multi_tick_advances_time(self, game_state: GameState) -> None:
        for _ in range(10):
            frame = game_tick(game_state, dt=0.1)
        assert game_state.tick_count == 10
        assert frame["sim_time"] > 0.0

    def test_frame_is_json_serializable(self, game_state: GameState) -> None:
        frame = game_tick(game_state, dt=0.1)
        # Must not raise
        payload = json.dumps(frame, default=str)
        assert len(payload) > 100

    def test_extended_simulation(self, game_state: GameState) -> None:
        """Run 50 ticks and verify the sim does not crash."""
        for _ in range(50):
            frame = game_tick(game_state, dt=0.1)
        assert game_state.tick_count == 50
        assert isinstance(frame, dict)
        # Some units should still be alive
        stats = frame["stats"]
        assert stats["total_units"] > 0

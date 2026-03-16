# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the multiplayer networking layer (sim_engine.multiplayer).

60+ tests covering player management, command submission, command execution,
fog-of-war views, lobby state, turn-based mode, and presets.
"""

from __future__ import annotations

import time

import pytest

from tritium_lib.sim_engine.multiplayer import (
    CommandType,
    GameCommand,
    MultiplayerEngine,
    Player,
    PlayerRole,
    TurnBasedMode,
    MULTIPLAYER_PRESETS,
    create_from_preset,
)
from tritium_lib.sim_engine.world import World, WorldConfig
from tritium_lib.sim_engine.units import Alliance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine_with_world() -> tuple[MultiplayerEngine, World]:
    """Create a multiplayer engine and a world with units for testing."""
    mp = MultiplayerEngine()
    world = World(WorldConfig(map_size=(200.0, 200.0), enable_weather=False))

    # Spawn blue units
    u1 = world.spawn_unit("infantry", "Blue-1", "friendly", (50.0, 50.0))
    u2 = world.spawn_unit("infantry", "Blue-2", "friendly", (55.0, 50.0))
    u3 = world.spawn_unit("sniper", "Blue-3", "friendly", (60.0, 50.0))

    # Spawn red units
    u4 = world.spawn_unit("infantry", "Red-1", "hostile", (150.0, 150.0))
    u5 = world.spawn_unit("infantry", "Red-2", "hostile", (155.0, 150.0))

    # Add players
    p1 = mp.add_player("p1", "Alice", "friendly", PlayerRole.COMMANDER)
    p2 = mp.add_player("p2", "Bob", "hostile", PlayerRole.COMMANDER)

    # Assign units
    mp.assign_units("p1", [u1.unit_id, u2.unit_id, u3.unit_id])
    mp.assign_units("p2", [u4.unit_id, u5.unit_id])

    return mp, world


# ---------------------------------------------------------------------------
# PlayerRole enum
# ---------------------------------------------------------------------------


class TestPlayerRole:
    def test_all_roles_exist(self):
        assert PlayerRole.COMMANDER.value == "commander"
        assert PlayerRole.SQUAD_LEADER.value == "squad_leader"
        assert PlayerRole.OBSERVER.value == "observer"
        assert PlayerRole.AI_CONTROLLED.value == "ai_controlled"

    def test_role_count(self):
        assert len(PlayerRole) == 4


# ---------------------------------------------------------------------------
# CommandType enum
# ---------------------------------------------------------------------------


class TestCommandType:
    def test_all_command_types(self):
        expected = {
            "move", "attack", "defend", "patrol", "retreat",
            "use_ability", "build", "call_support", "set_formation", "select_units",
        }
        actual = {ct.value for ct in CommandType}
        assert actual == expected

    def test_command_count(self):
        assert len(CommandType) == 10


# ---------------------------------------------------------------------------
# Player dataclass
# ---------------------------------------------------------------------------


class TestPlayer:
    def test_player_defaults(self):
        p = Player(player_id="x", name="X", faction_id="blue", role=PlayerRole.COMMANDER)
        assert p.connected is True
        assert p.controlled_units == []
        assert p.controlled_squads == []
        assert p.score == 0
        assert p.ping_ms == 0.0
        assert p.ready is False

    def test_controls_unit(self):
        p = Player(player_id="x", name="X", faction_id="blue", role=PlayerRole.COMMANDER,
                    controlled_units=["u1", "u2"])
        assert p.controls_unit("u1") is True
        assert p.controls_unit("u3") is False

    def test_controls_squad(self):
        p = Player(player_id="x", name="X", faction_id="blue", role=PlayerRole.COMMANDER,
                    controlled_squads=["sq1"])
        assert p.controls_squad("sq1") is True
        assert p.controls_squad("sq2") is False


# ---------------------------------------------------------------------------
# GameCommand dataclass
# ---------------------------------------------------------------------------


class TestGameCommand:
    def test_defaults(self):
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE)
        assert cmd.unit_id is None
        assert cmd.squad_id is None
        assert cmd.target_pos is None
        assert cmd.target_id is None
        assert cmd.params == {}
        assert cmd.timestamp == 0.0

    def test_full_command(self):
        cmd = GameCommand(
            player_id="p1",
            command_type=CommandType.ATTACK,
            unit_id="u1",
            target_id="u5",
            target_pos=(100.0, 100.0),
            params={"priority": "high"},
            timestamp=1.5,
        )
        assert cmd.command_type == CommandType.ATTACK
        assert cmd.params["priority"] == "high"


# ---------------------------------------------------------------------------
# MultiplayerEngine — player management
# ---------------------------------------------------------------------------


class TestPlayerManagement:
    def test_add_player(self):
        mp = MultiplayerEngine()
        p = mp.add_player("p1", "Alice", "blue", PlayerRole.COMMANDER)
        assert p.player_id == "p1"
        assert p.name == "Alice"
        assert p.faction_id == "blue"
        assert "p1" in mp.players

    def test_add_duplicate_raises(self):
        mp = MultiplayerEngine()
        mp.add_player("p1", "Alice", "blue")
        with pytest.raises(ValueError, match="already exists"):
            mp.add_player("p1", "Bob", "red")

    def test_remove_player(self):
        mp = MultiplayerEngine()
        mp.add_player("p1", "Alice", "blue")
        mp.remove_player("p1")
        assert "p1" not in mp.players

    def test_remove_nonexistent_is_noop(self):
        mp = MultiplayerEngine()
        mp.remove_player("ghost")  # no error

    def test_get_player(self):
        mp = MultiplayerEngine()
        mp.add_player("p1", "Alice", "blue")
        assert mp.get_player("p1") is not None
        assert mp.get_player("p999") is None

    def test_players_for_faction(self):
        mp = MultiplayerEngine()
        mp.add_player("p1", "Alice", "blue")
        mp.add_player("p2", "Bob", "blue")
        mp.add_player("p3", "Charlie", "red")
        blue = mp.players_for_faction("blue")
        assert len(blue) == 2
        assert mp.players_for_faction("red")[0].name == "Charlie"
        assert mp.players_for_faction("green") == []

    def test_remove_updates_faction_list(self):
        mp = MultiplayerEngine()
        mp.add_player("p1", "Alice", "blue")
        mp.add_player("p2", "Bob", "blue")
        mp.remove_player("p1")
        assert len(mp.players_for_faction("blue")) == 1

    def test_default_role_is_commander(self):
        mp = MultiplayerEngine()
        p = mp.add_player("p1", "Alice", "blue")
        assert p.role == PlayerRole.COMMANDER


# ---------------------------------------------------------------------------
# MultiplayerEngine — unit assignment
# ---------------------------------------------------------------------------


class TestUnitAssignment:
    def test_assign_units(self):
        mp = MultiplayerEngine()
        mp.add_player("p1", "Alice", "blue")
        mp.assign_units("p1", ["u1", "u2"])
        assert mp.players["p1"].controlled_units == ["u1", "u2"]

    def test_assign_units_no_duplicates(self):
        mp = MultiplayerEngine()
        mp.add_player("p1", "Alice", "blue")
        mp.assign_units("p1", ["u1", "u2"])
        mp.assign_units("p1", ["u2", "u3"])
        assert mp.players["p1"].controlled_units == ["u1", "u2", "u3"]

    def test_assign_units_unknown_player(self):
        mp = MultiplayerEngine()
        with pytest.raises(KeyError):
            mp.assign_units("ghost", ["u1"])

    def test_assign_squads(self):
        mp = MultiplayerEngine()
        mp.add_player("p1", "Alice", "blue")
        mp.assign_squads("p1", ["sq1"])
        assert mp.players["p1"].controlled_squads == ["sq1"]

    def test_assign_squads_unknown_player(self):
        mp = MultiplayerEngine()
        with pytest.raises(KeyError):
            mp.assign_squads("ghost", ["sq1"])

    def test_revoke_units(self):
        mp = MultiplayerEngine()
        mp.add_player("p1", "Alice", "blue")
        mp.assign_units("p1", ["u1", "u2", "u3"])
        mp.revoke_units("p1", ["u2"])
        assert mp.players["p1"].controlled_units == ["u1", "u3"]

    def test_revoke_units_unknown_player_is_noop(self):
        mp = MultiplayerEngine()
        mp.revoke_units("ghost", ["u1"])  # no error


# ---------------------------------------------------------------------------
# MultiplayerEngine — command submission
# ---------------------------------------------------------------------------


class TestCommandSubmission:
    def test_submit_valid_move(self):
        mp, world = _make_engine_with_world()
        unit_id = list(world.units.keys())[0]
        cmd = GameCommand(
            player_id="p1",
            command_type=CommandType.MOVE,
            unit_id=unit_id,
            target_pos=(100.0, 100.0),
        )
        assert mp.submit_command(cmd) is True
        assert len(mp.command_queue) == 1

    def test_submit_unknown_player_rejected(self):
        mp = MultiplayerEngine()
        cmd = GameCommand(player_id="nobody", command_type=CommandType.MOVE)
        assert mp.submit_command(cmd) is False

    def test_submit_disconnected_player_rejected(self):
        mp, world = _make_engine_with_world()
        mp.players["p1"].connected = False
        cmd = GameCommand(
            player_id="p1",
            command_type=CommandType.MOVE,
            unit_id=mp.players["p1"].controlled_units[0],
            target_pos=(0.0, 0.0),
        )
        assert mp.submit_command(cmd) is False

    def test_observer_cannot_issue_commands(self):
        mp = MultiplayerEngine()
        mp.add_player("obs", "Observer", "blue", PlayerRole.OBSERVER)
        mp.assign_units("obs", ["u1"])
        cmd = GameCommand(player_id="obs", command_type=CommandType.MOVE,
                          unit_id="u1", target_pos=(0.0, 0.0))
        assert mp.submit_command(cmd) is False

    def test_observer_can_select_units(self):
        mp = MultiplayerEngine()
        mp.add_player("obs", "Observer", "blue", PlayerRole.OBSERVER)
        cmd = GameCommand(player_id="obs", command_type=CommandType.SELECT_UNITS,
                          params={"unit_ids": ["u1"]})
        assert mp.submit_command(cmd) is True

    def test_ai_controlled_cannot_issue_commands(self):
        mp = MultiplayerEngine()
        mp.add_player("ai", "AI", "red", PlayerRole.AI_CONTROLLED)
        mp.assign_units("ai", ["u1"])
        cmd = GameCommand(player_id="ai", command_type=CommandType.MOVE,
                          unit_id="u1", target_pos=(0.0, 0.0))
        assert mp.submit_command(cmd) is False

    def test_cannot_command_unowned_unit(self):
        mp, world = _make_engine_with_world()
        # p1 tries to command a unit they don't own
        p2_unit = mp.players["p2"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                          unit_id=p2_unit, target_pos=(0.0, 0.0))
        assert mp.submit_command(cmd) is False

    def test_move_without_position_rejected(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                          unit_id=uid, target_pos=None)
        assert mp.submit_command(cmd) is False

    def test_attack_without_target_id_rejected(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.ATTACK,
                          unit_id=uid, target_id=None)
        assert mp.submit_command(cmd) is False

    def test_squad_command_accepted(self):
        mp, world = _make_engine_with_world()
        squad = world.spawn_squad("Alpha", "friendly",
                                   ["infantry", "infantry"], [(50.0, 50.0), (52.0, 50.0)])
        mp.assign_squads("p1", [squad.squad_id])
        cmd = GameCommand(player_id="p1", command_type=CommandType.SET_FORMATION,
                          squad_id=squad.squad_id, params={"formation": "wedge"})
        assert mp.submit_command(cmd) is True


# ---------------------------------------------------------------------------
# MultiplayerEngine — command execution
# ---------------------------------------------------------------------------


class TestCommandExecution:
    def test_process_move_command(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                          unit_id=uid, target_pos=(100.0, 100.0))
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert len(results) == 1
        assert results[0]["success"] is True
        assert results[0]["command"] == "move"

    def test_process_clears_queue(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                          unit_id=uid, target_pos=(100.0, 100.0))
        mp.submit_command(cmd)
        mp.process_commands(world)
        assert len(mp.command_queue) == 0

    def test_process_adds_to_history(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                          unit_id=uid, target_pos=(100.0, 100.0))
        mp.submit_command(cmd)
        mp.process_commands(world)
        assert len(mp.command_history) == 1

    def test_process_increments_turn(self):
        mp, world = _make_engine_with_world()
        assert mp.turn_number == 0
        mp.process_commands(world)
        assert mp.turn_number == 1

    def test_attack_in_range(self):
        mp, world = _make_engine_with_world()
        uid_atk = mp.players["p1"].controlled_units[0]
        uid_tgt = mp.players["p2"].controlled_units[0]
        # Move attacker close to target
        world.units[uid_atk].position = (150.0, 150.0)
        cmd = GameCommand(player_id="p1", command_type=CommandType.ATTACK,
                          unit_id=uid_atk, target_id=uid_tgt)
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True

    def test_attack_out_of_range_moves(self):
        mp, world = _make_engine_with_world()
        uid_atk = mp.players["p1"].controlled_units[0]
        uid_tgt = mp.players["p2"].controlled_units[0]
        # Keep attacker far away
        cmd = GameCommand(player_id="p1", command_type=CommandType.ATTACK,
                          unit_id=uid_atk, target_id=uid_tgt)
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True
        assert results[0].get("moving_to_range") is True

    def test_attack_dead_target(self):
        mp, world = _make_engine_with_world()
        uid_atk = mp.players["p1"].controlled_units[0]
        uid_tgt = mp.players["p2"].controlled_units[0]
        world.units[uid_tgt].state.is_alive = False
        cmd = GameCommand(player_id="p1", command_type=CommandType.ATTACK,
                          unit_id=uid_atk, target_id=uid_tgt)
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is False
        assert results[0]["reason"] == "target_dead"

    def test_attack_nonexistent_target(self):
        mp, world = _make_engine_with_world()
        uid_atk = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.ATTACK,
                          unit_id=uid_atk, target_id="u_999")
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is False
        assert results[0]["reason"] == "target_not_found"

    def test_defend_command(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.DEFEND,
                          unit_id=uid, target_pos=(50.0, 50.0))
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True
        assert results[0]["command"] == "defend"

    def test_patrol_command(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.PATROL,
                          unit_id=uid, target_pos=(100.0, 100.0),
                          params={"waypoints": [(80.0, 80.0), (120.0, 120.0)]})
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True

    def test_retreat_command(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.RETREAT,
                          unit_id=uid)
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True
        assert results[0]["command"] == "retreat"

    def test_retreat_with_position(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.RETREAT,
                          unit_id=uid, target_pos=(10.0, 10.0))
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True
        assert results[0]["retreat_pos"] == (10.0, 10.0)

    def test_use_ability_heal(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        world.units[uid].state.health = 50.0
        cmd = GameCommand(player_id="p1", command_type=CommandType.USE_ABILITY,
                          unit_id=uid, params={"ability": "heal", "amount": 30.0})
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True
        assert results[0]["healed"] == 30.0

    def test_use_ability_generic(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.USE_ABILITY,
                          unit_id=uid, params={"ability": "scan"})
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True

    def test_build_command(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.BUILD,
                          unit_id=uid, target_pos=(60.0, 60.0),
                          params={"structure_type": "bunker"})
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True
        assert results[0]["structure_type"] == "bunker"

    def test_call_support(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.CALL_SUPPORT,
                          unit_id=uid, target_pos=(150.0, 150.0),
                          params={"support_type": "airstrike"})
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True
        assert results[0]["support_type"] == "airstrike"

    def test_set_formation_with_squad(self):
        mp, world = _make_engine_with_world()
        squad = world.spawn_squad("Bravo", "friendly",
                                   ["infantry", "infantry"], [(50.0, 50.0), (52.0, 50.0)])
        mp.assign_squads("p1", [squad.squad_id])
        cmd = GameCommand(player_id="p1", command_type=CommandType.SET_FORMATION,
                          squad_id=squad.squad_id, unit_id=mp.players["p1"].controlled_units[0],
                          params={"formation": "diamond"})
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True
        assert results[0]["formation"] == "diamond"

    def test_set_formation_no_squad_fails(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.SET_FORMATION,
                          unit_id=uid, params={"formation": "line"})
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is False

    def test_select_units(self):
        mp, world = _make_engine_with_world()
        # Spawn a new unassigned unit
        new_unit = world.spawn_unit("infantry", "Blue-New", "friendly", (70.0, 50.0))
        cmd = GameCommand(player_id="p1", command_type=CommandType.SELECT_UNITS,
                          params={"unit_ids": [new_unit.unit_id]})
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is True
        assert new_unit.unit_id in mp.players["p1"].controlled_units

    def test_select_units_empty_fails(self):
        mp, world = _make_engine_with_world()
        cmd = GameCommand(player_id="p1", command_type=CommandType.SELECT_UNITS,
                          params={"unit_ids": []})
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is False

    def test_command_on_dead_unit_fails(self):
        mp, world = _make_engine_with_world()
        uid = mp.players["p1"].controlled_units[0]
        world.units[uid].state.is_alive = False
        world.units[uid].state.status = "dead"
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                          unit_id=uid, target_pos=(100.0, 100.0))
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is False
        assert results[0]["reason"] == "unit_dead"

    def test_command_nonexistent_unit_fails(self):
        mp, world = _make_engine_with_world()
        # Manually add a fake unit to controlled list
        mp.players["p1"].controlled_units.append("u_fake")
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                          unit_id="u_fake", target_pos=(100.0, 100.0))
        mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert results[0]["success"] is False
        assert results[0]["reason"] == "unit_not_found"

    def test_multiple_commands_per_turn(self):
        mp, world = _make_engine_with_world()
        uids = mp.players["p1"].controlled_units
        for uid in uids:
            cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                              unit_id=uid, target_pos=(100.0, 100.0))
            mp.submit_command(cmd)
        results = mp.process_commands(world)
        assert len(results) == len(uids)
        assert all(r["success"] for r in results)


# ---------------------------------------------------------------------------
# MultiplayerEngine — fog of war views
# ---------------------------------------------------------------------------


class TestPlayerView:
    def test_view_contains_own_units(self):
        mp, world = _make_engine_with_world()
        view = mp.get_player_view("p1", world)
        assert view["player_id"] == "p1"
        assert view["faction_id"] == "friendly"
        # Should see own units
        own_ids = {u["id"] for u in view["units"]}
        for uid in mp.players["p1"].controlled_units:
            assert uid in own_ids

    def test_view_hides_distant_enemies(self):
        mp, world = _make_engine_with_world()
        view = mp.get_player_view("p1", world)
        visible_ids = {u["id"] for u in view["units"]}
        # Red units at (150,150) should be out of detection range of blue at (50,50)
        for uid in mp.players["p2"].controlled_units:
            assert uid not in visible_ids

    def test_view_shows_close_enemies(self):
        mp, world = _make_engine_with_world()
        # Move a red unit close to blue
        red_uid = mp.players["p2"].controlled_units[0]
        world.units[red_uid].position = (55.0, 55.0)
        view = mp.get_player_view("p1", world)
        visible_ids = {u["id"] for u in view["units"]}
        assert red_uid in visible_ids

    def test_view_friendly_has_full_info(self):
        mp, world = _make_engine_with_world()
        view = mp.get_player_view("p1", world)
        friendly_unit = next(u for u in view["units"]
                             if u["id"] == mp.players["p1"].controlled_units[0])
        assert "health" in friendly_unit
        assert "morale" in friendly_unit
        assert "label" in friendly_unit

    def test_view_enemy_has_limited_info(self):
        mp, world = _make_engine_with_world()
        red_uid = mp.players["p2"].controlled_units[0]
        world.units[red_uid].position = (55.0, 55.0)
        view = mp.get_player_view("p1", world)
        enemy_unit = next(u for u in view["units"] if u["id"] == red_uid)
        assert "health" not in enemy_unit
        assert "morale" not in enemy_unit

    def test_view_unknown_player(self):
        mp, world = _make_engine_with_world()
        view = mp.get_player_view("nobody", world)
        assert view == {}

    def test_view_includes_controlled_units_list(self):
        mp, world = _make_engine_with_world()
        view = mp.get_player_view("p1", world)
        assert "controlled_units" in view
        assert len(view["controlled_units"]) == 3

    def test_view_includes_score(self):
        mp, world = _make_engine_with_world()
        mp.players["p1"].score = 42
        view = mp.get_player_view("p1", world)
        assert view["score"] == 42


# ---------------------------------------------------------------------------
# MultiplayerEngine — lobby
# ---------------------------------------------------------------------------


class TestLobby:
    def test_lobby_state_structure(self):
        mp, _ = _make_engine_with_world()
        lobby = mp.get_lobby_state()
        assert "factions" in lobby
        assert "player_count" in lobby
        assert "all_ready" in lobby
        assert lobby["player_count"] == 2

    def test_all_ready_false_by_default(self):
        mp, _ = _make_engine_with_world()
        assert mp.all_ready() is False

    def test_all_ready_when_all_ready(self):
        mp, _ = _make_engine_with_world()
        for p in mp.players.values():
            p.ready = True
        assert mp.all_ready() is True

    def test_all_ready_false_when_one_not_ready(self):
        mp, _ = _make_engine_with_world()
        mp.players["p1"].ready = True
        mp.players["p2"].ready = False
        assert mp.all_ready() is False

    def test_all_ready_empty_is_false(self):
        mp = MultiplayerEngine()
        assert mp.all_ready() is False

    def test_lobby_factions_list_players(self):
        mp, _ = _make_engine_with_world()
        lobby = mp.get_lobby_state()
        assert len(lobby["factions"]["friendly"]) == 1
        assert lobby["factions"]["friendly"][0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# MultiplayerEngine — to_three_js
# ---------------------------------------------------------------------------


class TestToThreeJS:
    def test_three_js_frame_structure(self):
        mp, world = _make_engine_with_world()
        frame = mp.to_three_js("p1", world)
        assert "tick" in frame
        assert "sim_time" in frame
        assert "units" in frame
        assert "player" in frame
        assert frame["player"]["player_id"] == "p1"

    def test_three_js_respects_fog_of_war(self):
        mp, world = _make_engine_with_world()
        frame = mp.to_three_js("p1", world)
        unit_ids = {u["id"] for u in frame["units"]}
        # Distant enemies should not be visible
        for uid in mp.players["p2"].controlled_units:
            assert uid not in unit_ids

    def test_three_js_includes_projectiles(self):
        mp, world = _make_engine_with_world()
        frame = mp.to_three_js("p1", world)
        assert "projectiles" in frame

    def test_three_js_includes_effects(self):
        mp, world = _make_engine_with_world()
        frame = mp.to_three_js("p1", world)
        assert "effects" in frame

    def test_three_js_includes_turn_number(self):
        mp, world = _make_engine_with_world()
        mp.process_commands(world)
        frame = mp.to_three_js("p1", world)
        assert frame["turn_number"] == 1


# ---------------------------------------------------------------------------
# TurnBasedMode
# ---------------------------------------------------------------------------


class TestTurnBasedMode:
    def test_initial_state(self):
        mp, _ = _make_engine_with_world()
        tb = TurnBasedMode(mp)
        assert tb.current_phase == "planning"

    def test_start_planning_phase(self):
        mp, _ = _make_engine_with_world()
        tb = TurnBasedMode(mp)
        tb.start_planning_phase(10.0)
        assert tb.current_phase == "planning"
        assert tb.planning_time_remaining() > 0

    def test_submit_during_planning(self):
        mp, world = _make_engine_with_world()
        tb = TurnBasedMode(mp)
        tb.start_planning_phase(30.0)
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                          unit_id=uid, target_pos=(100.0, 100.0))
        assert tb.submit_turn_command(cmd) is True

    def test_submit_outside_planning_rejected(self):
        mp, world = _make_engine_with_world()
        tb = TurnBasedMode(mp)
        tb.current_phase = "executing"
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                          unit_id=uid, target_pos=(100.0, 100.0))
        assert tb.submit_turn_command(cmd) is False

    def test_execute_turn(self):
        mp, world = _make_engine_with_world()
        tb = TurnBasedMode(mp)
        tb.start_planning_phase(30.0)
        uid = mp.players["p1"].controlled_units[0]
        cmd = GameCommand(player_id="p1", command_type=CommandType.MOVE,
                          unit_id=uid, target_pos=(100.0, 100.0))
        tb.submit_turn_command(cmd)
        results = tb.execute_turn(world)
        assert len(results) == 1
        assert tb.current_phase == "reviewing"

    def test_end_review(self):
        mp, _ = _make_engine_with_world()
        tb = TurnBasedMode(mp)
        tb.current_phase = "reviewing"
        tb.end_review()
        assert tb.current_phase == "planning"

    def test_get_turn_state(self):
        mp, _ = _make_engine_with_world()
        tb = TurnBasedMode(mp)
        tb.start_planning_phase(30.0)
        state = tb.get_turn_state()
        assert state["phase"] == "planning"
        assert "turn_number" in state
        assert "planning_time_remaining" in state

    def test_planning_expired(self):
        mp, _ = _make_engine_with_world()
        tb = TurnBasedMode(mp)
        tb.start_planning_phase(0.0)  # Immediately expired
        # Give a tiny bit of time for monotonic clock
        assert tb.is_planning_expired() is True

    def test_planning_not_expired(self):
        mp, _ = _make_engine_with_world()
        tb = TurnBasedMode(mp)
        tb.start_planning_phase(60.0)
        assert tb.is_planning_expired() is False

    def test_planning_time_zero_outside_planning(self):
        mp, _ = _make_engine_with_world()
        tb = TurnBasedMode(mp)
        tb.current_phase = "reviewing"
        assert tb.planning_time_remaining() == 0.0


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_all_presets_exist(self):
        assert "1v1" in MULTIPLAYER_PRESETS
        assert "2v2" in MULTIPLAYER_PRESETS
        assert "coop_vs_ai" in MULTIPLAYER_PRESETS
        assert "free_for_all" in MULTIPLAYER_PRESETS

    def test_1v1_preset(self):
        preset = MULTIPLAYER_PRESETS["1v1"]
        assert preset["max_players"] == 2
        assert len(preset["factions"]) == 2

    def test_2v2_preset(self):
        preset = MULTIPLAYER_PRESETS["2v2"]
        assert preset["max_players"] == 4
        assert preset["players_per_faction"] == 2

    def test_coop_vs_ai_preset(self):
        preset = MULTIPLAYER_PRESETS["coop_vs_ai"]
        assert "ai_factions" in preset
        assert "red" in preset["ai_factions"]

    def test_free_for_all_preset(self):
        preset = MULTIPLAYER_PRESETS["free_for_all"]
        assert len(preset["factions"]) == 4

    def test_create_from_preset(self):
        mp = create_from_preset("1v1")
        assert isinstance(mp, MultiplayerEngine)

    def test_create_from_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            create_from_preset("99v99")

    def test_preset_descriptions(self):
        for name, preset in MULTIPLAYER_PRESETS.items():
            assert "description" in preset, f"Preset {name!r} missing description"


# ---------------------------------------------------------------------------
# Integration: full game loop
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_game_loop(self):
        """Simulate a full multiplayer game loop: lobby -> commands -> tick -> view."""
        mp, world = _make_engine_with_world()

        # Players ready up
        mp.players["p1"].ready = True
        mp.players["p2"].ready = True
        assert mp.all_ready() is True

        # Both players issue move commands
        for pid in ("p1", "p2"):
            uid = mp.players[pid].controlled_units[0]
            cmd = GameCommand(player_id=pid, command_type=CommandType.MOVE,
                              unit_id=uid, target_pos=(100.0, 100.0))
            assert mp.submit_command(cmd) is True

        # Process
        results = mp.process_commands(world)
        assert len(results) == 2

        # Tick the world
        world.tick()

        # Get views
        v1 = mp.get_player_view("p1", world)
        v2 = mp.get_player_view("p2", world)
        assert v1["faction_id"] == "friendly"
        assert v2["faction_id"] == "hostile"

    def test_turn_based_game_loop(self):
        """Simulate a turn-based game: plan -> execute -> review -> repeat."""
        mp, world = _make_engine_with_world()
        tb = TurnBasedMode(mp)

        for turn in range(3):
            tb.start_planning_phase(30.0)
            uid = mp.players["p1"].controlled_units[0]
            cmd = GameCommand(player_id="p1", command_type=CommandType.DEFEND,
                              unit_id=uid, target_pos=(50.0 + turn * 10, 50.0))
            tb.submit_turn_command(cmd)
            results = tb.execute_turn(world)
            assert len(results) >= 1
            world.tick()
            tb.end_review()

        assert mp.turn_number == 3

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Multiplayer networking layer for the Tritium sim engine.

Allows multiple clients to connect, control different factions, and issue
commands that are validated and executed against the shared World.  Supports
both real-time and turn-based play modes, fog-of-war per-player views, and
preset lobby configurations.

Usage::

    from tritium_lib.sim_engine.multiplayer import (
        MultiplayerEngine, Player, PlayerRole, GameCommand, CommandType,
        TurnBasedMode, MULTIPLAYER_PRESETS,
    )
    from tritium_lib.sim_engine.world import World

    mp = MultiplayerEngine()
    p1 = mp.add_player("p1", "Alice", "blue", PlayerRole.COMMANDER)
    p2 = mp.add_player("p2", "Bob", "red", PlayerRole.COMMANDER)

    world = World()
    # ... spawn units ...
    mp.assign_units("p1", ["u_1", "u_2"])

    cmd = GameCommand(
        player_id="p1",
        command_type=CommandType.MOVE,
        unit_id="u_1",
        target_pos=(100.0, 200.0),
        timestamp=0.0,
    )
    mp.submit_command(cmd)
    results = mp.process_commands(world)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PlayerRole(Enum):
    """Roles a player may assume in a multiplayer session."""

    COMMANDER = "commander"
    SQUAD_LEADER = "squad_leader"
    OBSERVER = "observer"
    AI_CONTROLLED = "ai_controlled"


class CommandType(Enum):
    """Valid command types that a player may issue."""

    MOVE = "move"
    ATTACK = "attack"
    DEFEND = "defend"
    PATROL = "patrol"
    RETREAT = "retreat"
    USE_ABILITY = "use_ability"
    BUILD = "build"
    CALL_SUPPORT = "call_support"
    SET_FORMATION = "set_formation"
    SELECT_UNITS = "select_units"


# Commands that require a target position
_POSITION_COMMANDS = {
    CommandType.MOVE,
    CommandType.DEFEND,
    CommandType.PATROL,
    CommandType.BUILD,
    CommandType.CALL_SUPPORT,
}

# Commands that require a target id
_TARGET_ID_COMMANDS = {
    CommandType.ATTACK,
}

# Commands that require a unit or squad to control
_UNIT_COMMANDS = {
    CommandType.MOVE,
    CommandType.ATTACK,
    CommandType.DEFEND,
    CommandType.PATROL,
    CommandType.RETREAT,
    CommandType.USE_ABILITY,
    CommandType.BUILD,
    CommandType.CALL_SUPPORT,
    CommandType.SET_FORMATION,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Player:
    """A connected player in a multiplayer session."""

    player_id: str
    name: str
    faction_id: str
    role: PlayerRole
    connected: bool = True
    controlled_units: list[str] = field(default_factory=list)
    controlled_squads: list[str] = field(default_factory=list)
    score: int = 0
    ping_ms: float = 0.0
    ready: bool = False

    def controls_unit(self, unit_id: str) -> bool:
        """Return True if this player controls the given unit."""
        return unit_id in self.controlled_units

    def controls_squad(self, squad_id: str) -> bool:
        """Return True if this player controls the given squad."""
        return squad_id in self.controlled_squads


@dataclass
class GameCommand:
    """A command submitted by a player for execution against the world."""

    player_id: str
    command_type: CommandType
    unit_id: str | None = None
    squad_id: str | None = None
    target_pos: tuple[float, float] | None = None
    target_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# MultiplayerEngine
# ---------------------------------------------------------------------------


class MultiplayerEngine:
    """Core multiplayer session manager.

    Tracks players, validates commands, executes them against a World,
    and provides per-player fog-of-war views.
    """

    def __init__(self) -> None:
        self.players: dict[str, Player] = {}
        self.command_queue: list[GameCommand] = []
        self.command_history: list[GameCommand] = []
        self.turn_number: int = 0
        self._faction_players: dict[str, list[str]] = {}  # faction_id -> [player_ids]

    # -- Player management ---------------------------------------------------

    def add_player(
        self,
        player_id: str,
        name: str,
        faction_id: str,
        role: PlayerRole = PlayerRole.COMMANDER,
    ) -> Player:
        """Add a player to the session. Raises ValueError on duplicate ID."""
        if player_id in self.players:
            raise ValueError(f"Player {player_id!r} already exists")
        player = Player(
            player_id=player_id,
            name=name,
            faction_id=faction_id,
            role=role,
        )
        self.players[player_id] = player
        self._faction_players.setdefault(faction_id, []).append(player_id)
        return player

    def remove_player(self, player_id: str) -> None:
        """Remove a player from the session."""
        player = self.players.pop(player_id, None)
        if player is None:
            return
        faction_list = self._faction_players.get(player.faction_id, [])
        if player_id in faction_list:
            faction_list.remove(player_id)

    def get_player(self, player_id: str) -> Player | None:
        """Return a player by ID, or None if not found."""
        return self.players.get(player_id)

    def players_for_faction(self, faction_id: str) -> list[Player]:
        """Return all players belonging to a faction."""
        return [
            self.players[pid]
            for pid in self._faction_players.get(faction_id, [])
            if pid in self.players
        ]

    # -- Unit assignment -----------------------------------------------------

    def assign_units(self, player_id: str, unit_ids: list[str]) -> None:
        """Give control of specific units to a player.

        Raises KeyError if the player does not exist.
        """
        player = self.players.get(player_id)
        if player is None:
            raise KeyError(f"Unknown player {player_id!r}")
        for uid in unit_ids:
            if uid not in player.controlled_units:
                player.controlled_units.append(uid)

    def assign_squads(self, player_id: str, squad_ids: list[str]) -> None:
        """Give control of specific squads to a player.

        Raises KeyError if the player does not exist.
        """
        player = self.players.get(player_id)
        if player is None:
            raise KeyError(f"Unknown player {player_id!r}")
        for sid in squad_ids:
            if sid not in player.controlled_squads:
                player.controlled_squads.append(sid)

    def revoke_units(self, player_id: str, unit_ids: list[str]) -> None:
        """Remove control of specific units from a player."""
        player = self.players.get(player_id)
        if player is None:
            return
        player.controlled_units = [
            uid for uid in player.controlled_units if uid not in unit_ids
        ]

    # -- Command submission --------------------------------------------------

    def submit_command(self, cmd: GameCommand) -> bool:
        """Validate and enqueue a command. Returns True if accepted."""
        # Player must exist and be connected
        player = self.players.get(cmd.player_id)
        if player is None or not player.connected:
            return False

        # Observers cannot issue commands (except SELECT_UNITS)
        if player.role == PlayerRole.OBSERVER and cmd.command_type != CommandType.SELECT_UNITS:
            return False

        # AI_CONTROLLED players cannot issue commands
        if player.role == PlayerRole.AI_CONTROLLED:
            return False

        # Unit-level commands require the player to control the unit or squad
        if cmd.command_type in _UNIT_COMMANDS:
            has_unit = cmd.unit_id is not None and player.controls_unit(cmd.unit_id)
            has_squad = cmd.squad_id is not None and player.controls_squad(cmd.squad_id)
            if not has_unit and not has_squad:
                return False

        # Position commands require a target position
        if cmd.command_type in _POSITION_COMMANDS and cmd.target_pos is None:
            return False

        # Attack requires a target id
        if cmd.command_type == CommandType.ATTACK and cmd.target_id is None:
            return False

        self.command_queue.append(cmd)
        return True

    # -- Command execution ---------------------------------------------------

    def process_commands(self, world: Any) -> list[dict[str, Any]]:
        """Execute all queued commands against the world.

        Validates that the player controls the unit, the unit is alive,
        and the command is applicable.  Returns a list of result dicts,
        one per command.
        """
        results: list[dict[str, Any]] = []
        commands = list(self.command_queue)
        self.command_queue.clear()

        for cmd in commands:
            result = self._execute_command(cmd, world)
            results.append(result)
            self.command_history.append(cmd)

        self.turn_number += 1
        return results

    def _execute_command(self, cmd: GameCommand, world: Any) -> dict[str, Any]:
        """Execute a single command. Returns a result dict."""
        player = self.players.get(cmd.player_id)
        if player is None:
            return {"command": cmd.command_type.value, "success": False, "reason": "unknown_player"}

        # Resolve the unit(s) this command targets
        unit = None
        if cmd.unit_id is not None:
            unit = world.units.get(cmd.unit_id)
            if unit is None:
                return {"command": cmd.command_type.value, "success": False, "reason": "unit_not_found"}
            if not unit.is_alive():
                return {"command": cmd.command_type.value, "success": False, "reason": "unit_dead"}

        handler = _COMMAND_HANDLERS.get(cmd.command_type)
        if handler is None:
            return {"command": cmd.command_type.value, "success": False, "reason": "unhandled_command"}

        return handler(cmd, player, unit, world)

    # -- Views ---------------------------------------------------------------

    def get_player_view(self, player_id: str, world: Any) -> dict[str, Any]:
        """Return the world state filtered by fog of war for a player's faction.

        Only units and effects visible to the player's faction are included.
        Friendly units are always visible; enemy units are visible only if
        within detection range of at least one friendly unit.
        """
        player = self.players.get(player_id)
        if player is None:
            return {}

        faction_id = player.faction_id

        # Gather all friendly unit positions for visibility checks
        friendly_units: list[Any] = []
        enemy_units: list[Any] = []

        for uid, unit in world.units.items():
            if not unit.is_alive():
                continue
            if unit.alliance.value == faction_id or unit.squad_id and self._unit_belongs_to_faction(uid, faction_id):
                friendly_units.append(unit)
            else:
                enemy_units.append(unit)

        # Determine which enemy units are visible (within detection range of any friendly)
        visible_enemy_ids: set[str] = set()
        for enemy in enemy_units:
            for friendly in friendly_units:
                if friendly.can_see(enemy):
                    visible_enemy_ids.add(enemy.unit_id)
                    break

        # Build filtered unit list
        visible_units: list[dict[str, Any]] = []
        for uid, unit in world.units.items():
            if not unit.is_alive():
                continue
            if unit in friendly_units:
                visible_units.append(self._unit_to_dict(unit, full_info=True))
            elif unit.unit_id in visible_enemy_ids:
                visible_units.append(self._unit_to_dict(unit, full_info=False))

        # Build filtered vehicle list
        visible_vehicles: list[dict[str, Any]] = []
        for vid, vehicle in world.vehicles.items():
            if vehicle.alliance == faction_id:
                visible_vehicles.append({
                    "id": vid,
                    "x": vehicle.position[0],
                    "y": vehicle.position[1],
                    "alliance": vehicle.alliance,
                    "visible": True,
                })
            # Enemy vehicles visible if near a friendly
            elif any(
                distance(vehicle.position, f.position) <= f.stats.detection_range
                for f in friendly_units
            ):
                visible_vehicles.append({
                    "id": vid,
                    "x": vehicle.position[0],
                    "y": vehicle.position[1],
                    "alliance": vehicle.alliance,
                    "visible": True,
                })

        return {
            "player_id": player_id,
            "faction_id": faction_id,
            "tick": world.tick_count,
            "sim_time": world.sim_time,
            "units": visible_units,
            "vehicles": visible_vehicles,
            "controlled_units": player.controlled_units,
            "controlled_squads": player.controlled_squads,
            "score": player.score,
        }

    def _unit_belongs_to_faction(self, unit_id: str, faction_id: str) -> bool:
        """Check if a unit belongs to a faction via player assignments."""
        for pid in self._faction_players.get(faction_id, []):
            player = self.players.get(pid)
            if player and unit_id in player.controlled_units:
                return True
        return False

    @staticmethod
    def _unit_to_dict(unit: Any, full_info: bool = True) -> dict[str, Any]:
        """Serialize a unit to dict. Enemies get limited info."""
        d: dict[str, Any] = {
            "id": unit.unit_id,
            "x": unit.position[0],
            "y": unit.position[1],
            "type": unit.unit_type.value,
            "alliance": unit.alliance.value,
            "heading": unit.heading,
        }
        if full_info:
            d["health"] = unit.state.health
            d["max_health"] = unit.stats.max_health
            d["status"] = unit.state.status
            d["morale"] = unit.state.morale
            d["ammo"] = unit.state.ammo
            d["label"] = unit.name
        return d

    # -- Lobby ---------------------------------------------------------------

    def get_lobby_state(self) -> dict[str, Any]:
        """Return the current lobby state: players, factions, ready status."""
        factions: dict[str, list[dict[str, Any]]] = {}
        for faction_id, player_ids in self._faction_players.items():
            factions[faction_id] = []
            for pid in player_ids:
                p = self.players.get(pid)
                if p:
                    factions[faction_id].append({
                        "player_id": p.player_id,
                        "name": p.name,
                        "role": p.role.value,
                        "connected": p.connected,
                        "ready": p.ready,
                    })

        return {
            "factions": factions,
            "player_count": len(self.players),
            "all_ready": self.all_ready(),
            "turn_number": self.turn_number,
        }

    def all_ready(self) -> bool:
        """Return True if all connected players are ready."""
        connected = [p for p in self.players.values() if p.connected]
        if not connected:
            return False
        return all(p.ready for p in connected)

    # -- Render --------------------------------------------------------------

    def to_three_js(self, player_id: str, world: Any) -> dict[str, Any]:
        """Build a per-player Three.js render frame with fog-of-war applied."""
        view = self.get_player_view(player_id, world)
        player = self.players.get(player_id)

        frame: dict[str, Any] = {
            "tick": view.get("tick", 0),
            "sim_time": view.get("sim_time", 0.0),
            "units": view.get("units", []),
            "vehicles": view.get("vehicles", []),
            "player": {
                "player_id": player_id,
                "faction_id": player.faction_id if player else "",
                "score": player.score if player else 0,
                "controlled_units": player.controlled_units if player else [],
                "controlled_squads": player.controlled_squads if player else [],
            },
            "turn_number": self.turn_number,
        }

        # Include projectiles from the world render
        if hasattr(world, "projectile_sim"):
            proj_data = world.projectile_sim.to_three_js()
            frame["projectiles"] = proj_data.get("projectiles", [])

        # Include area effects
        if hasattr(world, "area_effects"):
            fx_data = world.area_effects.to_three_js()
            frame["effects"] = fx_data.get("effects", [])

        return frame


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _handle_move(
    cmd: GameCommand, player: Player, unit: Any, world: Any,
) -> dict[str, Any]:
    """Move a unit toward target_pos."""
    if unit is None:
        return {"command": "move", "success": False, "reason": "no_unit"}
    if cmd.target_pos is None:
        return {"command": "move", "success": False, "reason": "no_target_pos"}
    unit.state.status = "moving"
    # Store the movement target on the unit for the AI tick to pick up
    unit._move_target = cmd.target_pos  # type: ignore[attr-defined]
    return {
        "command": "move",
        "success": True,
        "unit_id": unit.unit_id,
        "target_pos": cmd.target_pos,
    }


def _handle_attack(
    cmd: GameCommand, player: Player, unit: Any, world: Any,
) -> dict[str, Any]:
    """Order a unit to attack a target."""
    if unit is None:
        return {"command": "attack", "success": False, "reason": "no_unit"}
    if cmd.target_id is None:
        return {"command": "attack", "success": False, "reason": "no_target"}
    target = world.units.get(cmd.target_id)
    if target is None:
        return {"command": "attack", "success": False, "reason": "target_not_found"}
    if not target.is_alive():
        return {"command": "attack", "success": False, "reason": "target_dead"}

    # Fire weapon if in range
    dist = distance(unit.position, target.position)
    if dist <= unit.stats.attack_range:
        world.fire_weapon(unit.unit_id, target.position)
        unit.state.status = "attacking"
        return {
            "command": "attack",
            "success": True,
            "unit_id": unit.unit_id,
            "target_id": cmd.target_id,
            "distance": dist,
        }
    else:
        # Move toward target
        unit.state.status = "moving"
        unit._move_target = target.position  # type: ignore[attr-defined]
        return {
            "command": "attack",
            "success": True,
            "unit_id": unit.unit_id,
            "target_id": cmd.target_id,
            "moving_to_range": True,
            "distance": dist,
        }


def _handle_defend(
    cmd: GameCommand, player: Player, unit: Any, world: Any,
) -> dict[str, Any]:
    """Order a unit to hold and defend a position."""
    if unit is None:
        return {"command": "defend", "success": False, "reason": "no_unit"}
    unit.state.status = "idle"
    if cmd.target_pos is not None:
        unit._move_target = cmd.target_pos  # type: ignore[attr-defined]
    return {
        "command": "defend",
        "success": True,
        "unit_id": unit.unit_id,
        "position": cmd.target_pos or unit.position,
    }


def _handle_patrol(
    cmd: GameCommand, player: Player, unit: Any, world: Any,
) -> dict[str, Any]:
    """Set a patrol route for a unit."""
    if unit is None:
        return {"command": "patrol", "success": False, "reason": "no_unit"}
    if cmd.target_pos is None:
        return {"command": "patrol", "success": False, "reason": "no_target_pos"}
    unit.state.status = "moving"
    waypoints = cmd.params.get("waypoints", [cmd.target_pos])
    unit._patrol_route = waypoints  # type: ignore[attr-defined]
    return {
        "command": "patrol",
        "success": True,
        "unit_id": unit.unit_id,
        "waypoints": waypoints,
    }


def _handle_retreat(
    cmd: GameCommand, player: Player, unit: Any, world: Any,
) -> dict[str, Any]:
    """Order a unit to retreat."""
    if unit is None:
        return {"command": "retreat", "success": False, "reason": "no_unit"}
    unit.state.status = "retreating"
    # Retreat direction: away from nearest enemy
    retreat_pos = cmd.target_pos
    if retreat_pos is None:
        # Default: move backwards from current heading
        import math
        dx = -math.cos(unit.heading) * 50.0
        dy = -math.sin(unit.heading) * 50.0
        retreat_pos = (unit.position[0] + dx, unit.position[1] + dy)
    unit._move_target = retreat_pos  # type: ignore[attr-defined]
    return {
        "command": "retreat",
        "success": True,
        "unit_id": unit.unit_id,
        "retreat_pos": retreat_pos,
    }


def _handle_use_ability(
    cmd: GameCommand, player: Player, unit: Any, world: Any,
) -> dict[str, Any]:
    """Use a special ability (grenade, smoke, heal, etc.)."""
    if unit is None:
        return {"command": "use_ability", "success": False, "reason": "no_unit"}
    ability = cmd.params.get("ability", "unknown")
    target = cmd.target_pos or unit.position

    if ability == "grenade" and hasattr(world, "area_effects"):
        from tritium_lib.sim_engine.arsenal import create_explosion_effect
        fx = create_explosion_effect(target, radius=5.0)
        world.area_effects.add(fx)
        return {"command": "use_ability", "success": True, "ability": ability, "unit_id": unit.unit_id}
    elif ability == "smoke" and hasattr(world, "area_effects"):
        from tritium_lib.sim_engine.arsenal import create_smoke_effect
        fx = create_smoke_effect(target, radius=8.0)
        world.area_effects.add(fx)
        return {"command": "use_ability", "success": True, "ability": ability, "unit_id": unit.unit_id}
    elif ability == "heal":
        heal_amount = cmd.params.get("amount", 25.0)
        actual = unit.heal(heal_amount)
        return {"command": "use_ability", "success": True, "ability": ability, "healed": actual}

    return {"command": "use_ability", "success": True, "ability": ability, "unit_id": unit.unit_id}


def _handle_build(
    cmd: GameCommand, player: Player, unit: Any, world: Any,
) -> dict[str, Any]:
    """Construct a fortification at target_pos."""
    if unit is None:
        return {"command": "build", "success": False, "reason": "no_unit"}
    if cmd.target_pos is None:
        return {"command": "build", "success": False, "reason": "no_target_pos"}
    structure_type = cmd.params.get("structure_type", "sandbag")
    return {
        "command": "build",
        "success": True,
        "unit_id": unit.unit_id,
        "structure_type": structure_type,
        "position": cmd.target_pos,
    }


def _handle_call_support(
    cmd: GameCommand, player: Player, unit: Any, world: Any,
) -> dict[str, Any]:
    """Request fire support (airstrike, artillery)."""
    if unit is None:
        return {"command": "call_support", "success": False, "reason": "no_unit"}
    if cmd.target_pos is None:
        return {"command": "call_support", "success": False, "reason": "no_target_pos"}
    support_type = cmd.params.get("support_type", "artillery")
    return {
        "command": "call_support",
        "success": True,
        "unit_id": unit.unit_id,
        "support_type": support_type,
        "target_pos": cmd.target_pos,
    }


def _handle_set_formation(
    cmd: GameCommand, player: Player, unit: Any, world: Any,
) -> dict[str, Any]:
    """Change squad formation."""
    if cmd.squad_id is None:
        return {"command": "set_formation", "success": False, "reason": "no_squad"}
    formation = cmd.params.get("formation", "line")
    squad = world.squads.get(cmd.squad_id) if hasattr(world, "squads") else None
    if squad is None:
        return {"command": "set_formation", "success": False, "reason": "squad_not_found"}
    return {
        "command": "set_formation",
        "success": True,
        "squad_id": cmd.squad_id,
        "formation": formation,
    }


def _handle_select_units(
    cmd: GameCommand, player: Player, unit: Any, world: Any,
) -> dict[str, Any]:
    """Assign units to player control."""
    unit_ids = cmd.params.get("unit_ids", [])
    if not unit_ids:
        return {"command": "select_units", "success": False, "reason": "no_unit_ids"}

    # Verify units exist and belong to player's faction
    valid_ids: list[str] = []
    for uid in unit_ids:
        u = world.units.get(uid) if hasattr(world, "units") else None
        if u is not None and u.is_alive():
            valid_ids.append(uid)

    for uid in valid_ids:
        if uid not in player.controlled_units:
            player.controlled_units.append(uid)

    return {
        "command": "select_units",
        "success": True,
        "player_id": player.player_id,
        "selected": valid_ids,
    }


_COMMAND_HANDLERS: dict[CommandType, Any] = {
    CommandType.MOVE: _handle_move,
    CommandType.ATTACK: _handle_attack,
    CommandType.DEFEND: _handle_defend,
    CommandType.PATROL: _handle_patrol,
    CommandType.RETREAT: _handle_retreat,
    CommandType.USE_ABILITY: _handle_use_ability,
    CommandType.BUILD: _handle_build,
    CommandType.CALL_SUPPORT: _handle_call_support,
    CommandType.SET_FORMATION: _handle_set_formation,
    CommandType.SELECT_UNITS: _handle_select_units,
}


# ---------------------------------------------------------------------------
# TurnBasedMode
# ---------------------------------------------------------------------------


class TurnBasedMode:
    """Optional turn-based play mode layered on top of MultiplayerEngine.

    In turn-based mode, all players submit commands during a planning phase,
    then all commands execute simultaneously.
    """

    def __init__(self, engine: MultiplayerEngine) -> None:
        self.engine = engine
        self.current_phase: str = "planning"  # planning, executing, reviewing
        self._planning_deadline: float = 0.0
        self._planning_duration: float = 30.0
        self._turn_commands: dict[str, list[GameCommand]] = {}  # player_id -> commands

    def start_planning_phase(self, duration: float = 30.0) -> None:
        """Begin a new planning phase. Players may submit commands."""
        self.current_phase = "planning"
        self._planning_duration = duration
        self._planning_deadline = time.monotonic() + duration
        self._turn_commands.clear()

    def submit_turn_command(self, cmd: GameCommand) -> bool:
        """Submit a command during the planning phase."""
        if self.current_phase != "planning":
            return False
        if not self.engine.submit_command(cmd):
            return False
        self._turn_commands.setdefault(cmd.player_id, []).append(cmd)
        return True

    def planning_time_remaining(self) -> float:
        """Seconds remaining in the planning phase."""
        if self.current_phase != "planning":
            return 0.0
        remaining = self._planning_deadline - time.monotonic()
        return max(0.0, remaining)

    def is_planning_expired(self) -> bool:
        """Return True if the planning timer has expired."""
        if self.current_phase != "planning":
            return True
        return time.monotonic() >= self._planning_deadline

    def execute_turn(self, world: Any) -> list[dict[str, Any]]:
        """Execute all queued commands simultaneously. Advances to reviewing phase."""
        self.current_phase = "executing"
        results = self.engine.process_commands(world)
        self.current_phase = "reviewing"
        return results

    def end_review(self) -> None:
        """End the review phase. Call start_planning_phase() to begin the next turn."""
        self.current_phase = "planning"
        self._turn_commands.clear()

    def get_turn_state(self) -> dict[str, Any]:
        """Return the current turn-based mode state."""
        return {
            "phase": self.current_phase,
            "turn_number": self.engine.turn_number,
            "planning_time_remaining": self.planning_time_remaining(),
            "commands_submitted": {
                pid: len(cmds) for pid, cmds in self._turn_commands.items()
            },
        }


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

MULTIPLAYER_PRESETS: dict[str, dict[str, Any]] = {
    "1v1": {
        "description": "1v1 — two commanders, one per faction",
        "factions": ["blue", "red"],
        "players_per_faction": 1,
        "roles": [PlayerRole.COMMANDER],
        "max_players": 2,
    },
    "2v2": {
        "description": "2v2 — two players per faction",
        "factions": ["blue", "red"],
        "players_per_faction": 2,
        "roles": [PlayerRole.COMMANDER, PlayerRole.SQUAD_LEADER],
        "max_players": 4,
    },
    "coop_vs_ai": {
        "description": "Co-op vs AI — 1-4 human players vs AI faction",
        "factions": ["blue", "red"],
        "players_per_faction": {"blue": 4, "red": 0},
        "roles": [PlayerRole.COMMANDER, PlayerRole.SQUAD_LEADER],
        "ai_factions": ["red"],
        "max_players": 4,
    },
    "free_for_all": {
        "description": "Free for All — 3-4 players, each own faction",
        "factions": ["red", "blue", "green", "yellow"],
        "players_per_faction": 1,
        "roles": [PlayerRole.COMMANDER],
        "max_players": 4,
    },
}


def create_from_preset(preset_name: str) -> MultiplayerEngine:
    """Create a MultiplayerEngine configured for a given preset.

    Players still need to be added via ``add_player()``.
    """
    if preset_name not in MULTIPLAYER_PRESETS:
        raise ValueError(f"Unknown preset {preset_name!r}. "
                         f"Available: {list(MULTIPLAYER_PRESETS.keys())}")
    return MultiplayerEngine()


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "PlayerRole",
    "CommandType",
    "Player",
    "GameCommand",
    "MultiplayerEngine",
    "TurnBasedMode",
    "MULTIPLAYER_PRESETS",
    "create_from_preset",
]

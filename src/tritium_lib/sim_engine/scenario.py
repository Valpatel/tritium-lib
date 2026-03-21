# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Scenario tick runner — the main simulation game loop.

Runs a complete combat simulation tick-by-tick, managing wave spawning,
unit AI, combat resolution, objective tracking, and game-over conditions.

Usage::

    from tritium_lib.sim_engine.scenario import Scenario, ScenarioConfig, PRESET_SCENARIOS

    config = PRESET_SCENARIOS["skirmish"]
    sim = Scenario(config)
    sim.on("unit_killed", lambda e: print(f"Kill at tick {e.tick}"))
    sim.start()
    final_state = sim.run()
    print(sim.stats())
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from tritium_lib.sim_engine.ai.steering import (
    Vec2,
    distance,
    _sub,
    _add,
    _scale,
    seek,
    flee,
)
from tritium_lib.sim_engine.units import (
    Unit,
    Alliance,
    create_unit,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SimEvent:
    """A single discrete event emitted during simulation."""

    tick: int
    time: float
    event_type: str  # unit_spawned, unit_killed, damage_dealt, order_issued,
                     # wave_start, wave_end, objective_complete, scenario_end
    data: dict = field(default_factory=dict)


@dataclass
class WaveConfig:
    """Configuration for a single wave of hostiles."""

    wave_number: int
    spawn_delay: float = 2.0  # seconds between individual spawns
    hostiles: list[dict] = field(default_factory=list)
    # Each dict: {"template": "infantry", "count": 5,
    #             "spawn_pos": (x, y), "target_pos": (x, y)}
    wave_bonus: float = 0.0  # stat multiplier per wave (0.1 = 10% harder)


@dataclass
class Objective:
    """A win/loss condition for the scenario."""

    objective_type: str  # eliminate_all, survive_time, defend_point,
                         # reach_point, kill_count
    target_value: float  # time in seconds, kill count, etc.
    current_value: float = 0.0
    completed: bool = False


@dataclass
class ScenarioConfig:
    """Full scenario definition."""

    name: str
    description: str = ""
    tick_rate: float = 10.0  # ticks per second
    max_ticks: int = 6000    # 10 min at 10 tps
    waves: list[WaveConfig] = field(default_factory=list)
    objectives: list[Objective] = field(default_factory=list)
    friendly_units: list[dict] = field(default_factory=list)
    # Same format: {"template": "infantry", "count": 3,
    #               "spawn_pos": (x, y), "target_pos": (x, y)}
    map_size: tuple[float, float] = (200.0, 200.0)


@dataclass
class SimState:
    """Mutable simulation state — snapshot of the entire sim at a point in time."""

    tick: int = 0
    time: float = 0.0
    phase: str = "setup"  # setup, countdown, active, paused, game_over
    current_wave: int = 0
    units: dict[str, dict] = field(default_factory=dict)
    events: list[SimEvent] = field(default_factory=list)
    score: int = 0
    result: str = ""  # victory, defeat, draw


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


class Scenario:
    """The main simulation runner.

    Manages the full tick loop: spawning, AI, combat, objectives, events.
    """

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self.state = SimState()
        self._event_listeners: dict[str, list[Callable]] = {}

        # Live unit objects (not serialized in state.units until snapshot)
        self._units: dict[str, Unit] = {}
        self._next_unit_id = 0

        # Wave tracking
        self._wave_spawn_queue: list[tuple[float, str, Alliance, Vec2, Vec2]] = []
        # (spawn_time, template, alliance, spawn_pos, target_pos)
        self._wave_hostiles_alive: int = 0
        self._wave_started: bool = False
        self._all_waves_done: bool = False

        # Stats tracking
        self._friendly_kills: int = 0
        self._hostile_kills: int = 0
        self._total_damage_dealt: float = 0.0
        self._total_damage_taken: float = 0.0
        self._shots_fired: int = 0
        self._shots_hit: int = 0

    # -- Event system -------------------------------------------------------

    def on(self, event_type: str, callback: Callable) -> None:
        """Register an event listener."""
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(callback)

    def emit(self, event: SimEvent) -> None:
        """Emit an event to all registered listeners and record it."""
        self.state.events.append(event)
        for cb in self._event_listeners.get(event.event_type, []):
            cb(event)

    # -- Phase control ------------------------------------------------------

    def start(self) -> None:
        """Transition from setup to active, spawn friendly units."""
        if self.state.phase != "setup":
            return
        self.state.phase = "active"
        self.state.current_wave = 0
        self._spawn_friendlies()
        self._begin_wave(0)

    def pause(self) -> None:
        """Pause the simulation."""
        if self.state.phase == "active":
            self.state.phase = "paused"

    def resume(self) -> None:
        """Resume from pause."""
        if self.state.phase == "paused":
            self.state.phase = "active"

    # -- Tick ---------------------------------------------------------------

    def tick(self, dt: float | None = None) -> None:
        """Advance one simulation tick.

        Steps:
        1. Check wave spawning
        2. Unit AI decisions
        3. Resolve attacks
        4. Apply damage
        5. Morale/suppression decay
        6. Check objectives
        7. Check wave completion
        8. Check game over
        9. Emit events
        """
        if self.state.phase != "active":
            return

        if dt is None:
            dt = 1.0 / self.config.tick_rate

        self.state.tick += 1
        self.state.time += dt

        # 1. Process spawn queue
        self._process_spawns()

        # 2-4. Unit AI and combat
        self._update_units(dt)

        # 5. Morale/suppression recovery
        self._recover_units(dt)

        # 6. Check objectives
        self._update_objectives()

        # 7. Wave completion
        self._check_wave_completion()

        # 8. Game over
        self._check_game_over()

    def run(self, max_ticks: int | None = None) -> SimState:
        """Run the scenario to completion and return final state."""
        if self.state.phase == "setup":
            self.start()

        limit = max_ticks if max_ticks is not None else self.config.max_ticks

        while self.state.phase == "active" and self.state.tick < limit:
            self.tick()

        # If we hit the tick limit while still active, it's a draw/timeout
        if self.state.phase == "active":
            self.state.phase = "game_over"
            if not self.state.result:
                self.state.result = "draw"
            self.emit(SimEvent(
                tick=self.state.tick,
                time=self.state.time,
                event_type="scenario_end",
                data={"result": self.state.result, "reason": "tick_limit"},
            ))

        self._sync_units_to_state()
        return self.state

    def snapshot(self) -> dict:
        """Return a JSON-serializable state snapshot."""
        self._sync_units_to_state()
        return {
            "tick": self.state.tick,
            "time": round(self.state.time, 3),
            "phase": self.state.phase,
            "current_wave": self.state.current_wave,
            "units": dict(self.state.units),
            "event_count": len(self.state.events),
            "score": self.state.score,
            "result": self.state.result,
            "map_size": list(self.config.map_size),
            "scenario_name": self.config.name,
        }

    def stats(self) -> dict:
        """End-of-game statistics."""
        # Find MVP (most kills among friendlies)
        mvp_id = ""
        mvp_kills = 0
        for uid, unit in self._units.items():
            if unit.alliance == Alliance.FRIENDLY and unit.state.kill_count > mvp_kills:
                mvp_kills = unit.state.kill_count
                mvp_id = uid

        alive_friendly = sum(
            1 for u in self._units.values()
            if u.alliance == Alliance.FRIENDLY and u.state.is_alive
        )
        alive_hostile = sum(
            1 for u in self._units.values()
            if u.alliance == Alliance.HOSTILE and u.state.is_alive
        )

        return {
            "scenario": self.config.name,
            "result": self.state.result,
            "ticks": self.state.tick,
            "time": round(self.state.time, 2),
            "waves_cleared": self.state.current_wave,
            "friendly_kills": self._friendly_kills,
            "hostile_kills": self._hostile_kills,
            "total_damage_dealt": round(self._total_damage_dealt, 1),
            "total_damage_taken": round(self._total_damage_taken, 1),
            "shots_fired": self._shots_fired,
            "shots_hit": self._shots_hit,
            "accuracy": round(self._shots_hit / max(self._shots_fired, 1), 3),
            "score": self.state.score,
            "mvp": mvp_id,
            "mvp_kills": mvp_kills,
            "alive_friendly": alive_friendly,
            "alive_hostile": alive_hostile,
        }

    # -- Internal: spawning -------------------------------------------------

    def _gen_unit_id(self, prefix: str = "u") -> str:
        self._next_unit_id += 1
        return f"{prefix}_{self._next_unit_id:04d}"

    def _spawn_friendlies(self) -> None:
        """Spawn all friendly units immediately."""
        for group in self.config.friendly_units:
            template = group.get("template", "infantry")
            count = group.get("count", 1)
            spawn_pos = tuple(group.get("spawn_pos", (100.0, 100.0)))
            for i in range(count):
                uid = self._gen_unit_id("friendly")
                # Spread units slightly, clamped to map
                mx, my = self.config.map_size
                pos = (
                    max(0.0, min(mx, spawn_pos[0] + random.uniform(-5, 5))),
                    max(0.0, min(my, spawn_pos[1] + random.uniform(-5, 5))),
                )
                unit = create_unit(template, uid, f"F-{uid}", Alliance.FRIENDLY, pos)
                self._units[uid] = unit
                self.emit(SimEvent(
                    tick=self.state.tick,
                    time=self.state.time,
                    event_type="unit_spawned",
                    data={"unit_id": uid, "alliance": "friendly", "template": template},
                ))

    def _begin_wave(self, wave_idx: int) -> None:
        """Queue spawns for a wave."""
        if wave_idx >= len(self.config.waves):
            self._all_waves_done = True
            return

        wave = self.config.waves[wave_idx]
        self.state.current_wave = wave.wave_number
        self._wave_started = True
        self._wave_hostiles_alive = 0

        self.emit(SimEvent(
            tick=self.state.tick,
            time=self.state.time,
            event_type="wave_start",
            data={"wave": wave.wave_number},
        ))

        spawn_time = self.state.time
        for group in wave.hostiles:
            template = group.get("template", "infantry")
            count = group.get("count", 1)
            spawn_pos = tuple(group.get("spawn_pos", (0.0, 0.0)))
            target_pos = tuple(group.get("target_pos", (100.0, 100.0)))
            for i in range(count):
                self._wave_spawn_queue.append(
                    (spawn_time, template, Alliance.HOSTILE, spawn_pos, target_pos)
                )
                spawn_time += wave.spawn_delay

        # Apply wave bonus to queued spawns (stored as metadata)
        self._current_wave_bonus = wave.wave_bonus

    def _process_spawns(self) -> None:
        """Spawn units whose time has come."""
        remaining = []
        for spawn_time, template, alliance, spawn_pos, target_pos in self._wave_spawn_queue:
            if self.state.time >= spawn_time:
                uid = self._gen_unit_id("hostile" if alliance == Alliance.HOSTILE else "unit")
                pos = (
                    spawn_pos[0] + random.uniform(-3, 3),
                    spawn_pos[1] + random.uniform(-3, 3),
                )
                unit = create_unit(template, uid, f"H-{uid}", alliance, pos)

                # Apply wave bonus
                bonus = getattr(self, "_current_wave_bonus", 0.0)
                if bonus > 0:
                    unit.stats.max_health *= (1.0 + bonus)
                    unit.state.health = unit.stats.max_health
                    unit.stats.attack_damage *= (1.0 + bonus)
                    unit.stats.speed *= (1.0 + bonus * 0.5)

                # Store target position for AI
                unit._target_pos = target_pos  # type: ignore[attr-defined]
                self._units[uid] = unit
                self._wave_hostiles_alive += 1

                self.emit(SimEvent(
                    tick=self.state.tick,
                    time=self.state.time,
                    event_type="unit_spawned",
                    data={"unit_id": uid, "alliance": alliance.value, "template": template},
                ))
            else:
                remaining.append((spawn_time, template, alliance, spawn_pos, target_pos))
        self._wave_spawn_queue = remaining

    # -- Internal: AI and combat --------------------------------------------

    def _update_units(self, dt: float) -> None:
        """Run AI and combat for all alive units."""
        # Build target lists
        friendlies = [u for u in self._units.values()
                      if u.alliance == Alliance.FRIENDLY and u.state.is_alive]
        hostiles = [u for u in self._units.values()
                    if u.alliance == Alliance.HOSTILE and u.state.is_alive]

        # Process each alive unit
        for unit in list(self._units.values()):
            if not unit.state.is_alive:
                continue

            if unit.alliance == Alliance.HOSTILE:
                self._ai_hostile(unit, friendlies, dt)
            elif unit.alliance == Alliance.FRIENDLY:
                self._ai_friendly(unit, hostiles, dt)

    def _ai_hostile(self, unit: Unit, enemies: list[Unit], dt: float) -> None:
        """Simple hostile AI: pursue nearest enemy, attack in range."""
        if not enemies:
            # Move toward target position if no enemies
            target_pos = getattr(unit, "_target_pos", None)
            if target_pos:
                self._move_toward(unit, target_pos, dt)
            return

        # Find nearest enemy
        nearest = min(enemies, key=lambda e: unit.distance_to(e))
        dist = unit.distance_to(nearest)

        # Low morale: flee
        if unit.state.morale < 0.2:
            self._move_away_from(unit, nearest.position, dt)
            unit.state.status = "retreating"
            return

        # In attack range: shoot
        if dist <= unit.stats.attack_range:
            self._try_attack(unit, nearest, dt)
            unit.state.status = "attacking"
        else:
            # Move toward nearest enemy
            self._move_toward(unit, nearest.position, dt)
            unit.state.status = "moving"

    def _ai_friendly(self, unit: Unit, enemies: list[Unit], dt: float) -> None:
        """Simple friendly AI: engage nearest enemy, hold position if none."""
        if not enemies:
            unit.state.status = "idle"
            return

        nearest = min(enemies, key=lambda e: unit.distance_to(e))
        dist = unit.distance_to(nearest)

        # Low morale: retreat
        if unit.state.morale < 0.2:
            self._move_away_from(unit, nearest.position, dt)
            unit.state.status = "retreating"
            return

        if dist <= unit.stats.attack_range:
            self._try_attack(unit, nearest, dt)
            unit.state.status = "attacking"
        elif dist <= unit.stats.detection_range:
            self._move_toward(unit, nearest.position, dt)
            unit.state.status = "moving"
        else:
            unit.state.status = "idle"

    def _move_toward(self, unit: Unit, target: Vec2, dt: float) -> None:
        """Move a unit toward a target position."""
        direction = seek(unit.position, target, unit.effective_speed())
        new_pos = _add(unit.position, _scale(direction, dt))
        # Clamp to map bounds
        mx, my = self.config.map_size
        new_pos = (
            max(0.0, min(mx, new_pos[0])),
            max(0.0, min(my, new_pos[1])),
        )
        unit.position = new_pos

    def _move_away_from(self, unit: Unit, threat: Vec2, dt: float) -> None:
        """Move a unit away from a threat."""
        direction = flee(unit.position, threat, unit.effective_speed())
        new_pos = _add(unit.position, _scale(direction, dt))
        mx, my = self.config.map_size
        new_pos = (
            max(0.0, min(mx, new_pos[0])),
            max(0.0, min(my, new_pos[1])),
        )
        unit.position = new_pos

    def _try_attack(self, unit: Unit, target: Unit, dt: float) -> None:
        """Attempt to attack a target if cooldown allows."""
        if not unit.can_attack(self.state.time):
            return

        self._shots_fired += 1

        # Hit probability based on accuracy, distance, and suppression
        dist = unit.distance_to(target)
        range_factor = max(0.0, 1.0 - (dist / (unit.stats.attack_range * 1.5)))
        hit_chance = unit.effective_accuracy() * range_factor

        unit.state.last_attack_time = self.state.time

        if random.random() < hit_chance:
            self._shots_hit += 1
            actual_damage = target.take_damage(unit.stats.attack_damage)
            unit.state.damage_dealt += actual_damage

            if unit.alliance == Alliance.FRIENDLY:
                self._total_damage_dealt += actual_damage
            else:
                self._total_damage_taken += actual_damage

            # Suppression on target
            target.apply_suppression(0.15)

            # Reduce morale of nearby allies of the target
            for other in self._units.values():
                if (other.alliance == target.alliance
                        and other.state.is_alive
                        and other.unit_id != target.unit_id
                        and distance(other.position, target.position) < 15.0):
                    other.state.morale = max(0.0, other.state.morale - 0.02)

            self.emit(SimEvent(
                tick=self.state.tick,
                time=self.state.time,
                event_type="damage_dealt",
                data={
                    "attacker": unit.unit_id,
                    "target": target.unit_id,
                    "damage": round(actual_damage, 1),
                    "target_health": round(target.state.health, 1),
                },
            ))

            # Check kill
            if not target.state.is_alive:
                unit.state.kill_count += 1
                if target.alliance == Alliance.HOSTILE:
                    self._friendly_kills += 1
                    self._wave_hostiles_alive = max(0, self._wave_hostiles_alive - 1)
                    self.state.score += 100
                else:
                    self._hostile_kills += 1

                self.emit(SimEvent(
                    tick=self.state.tick,
                    time=self.state.time,
                    event_type="unit_killed",
                    data={
                        "killed": target.unit_id,
                        "killer": unit.unit_id,
                        "alliance": target.alliance.value,
                    },
                ))

    # -- Internal: recovery -------------------------------------------------

    def _recover_units(self, dt: float) -> None:
        """Decay suppression and recover morale for all alive units."""
        for unit in self._units.values():
            if not unit.state.is_alive:
                continue
            unit.recover_suppression(dt)
            # Morale recovery
            if unit.state.morale < 1.0:
                unit.state.morale = min(1.0, unit.state.morale + 0.01 * dt)

    # -- Internal: objectives -----------------------------------------------

    def _update_objectives(self) -> None:
        """Check and update objective progress."""
        for obj in self.config.objectives:
            if obj.completed:
                continue

            if obj.objective_type == "eliminate_all":
                alive_hostiles = sum(
                    1 for u in self._units.values()
                    if u.alliance == Alliance.HOSTILE and u.state.is_alive
                )
                pending_spawns = len(self._wave_spawn_queue)
                remaining_waves = (
                    len(self.config.waves)
                    - (self._get_wave_index() + 1)
                    if not self._all_waves_done else 0
                )
                if alive_hostiles == 0 and pending_spawns == 0 and self._all_waves_done:
                    obj.current_value = obj.target_value
                    obj.completed = True

            elif obj.objective_type == "survive_time":
                obj.current_value = self.state.time
                if self.state.time >= obj.target_value:
                    obj.completed = True

            elif obj.objective_type == "kill_count":
                obj.current_value = self._friendly_kills
                if self._friendly_kills >= obj.target_value:
                    obj.completed = True

            elif obj.objective_type == "defend_point":
                # Defend: check no hostiles within 20m of the objective point
                # target_value used as radius, current_value tracks time defended
                obj.current_value = self.state.time
                if self.state.time >= obj.target_value:
                    obj.completed = True

            if obj.completed:
                self.emit(SimEvent(
                    tick=self.state.tick,
                    time=self.state.time,
                    event_type="objective_complete",
                    data={"objective_type": obj.objective_type},
                ))

    def _get_wave_index(self) -> int:
        """Get the current wave index (0-based)."""
        for i, w in enumerate(self.config.waves):
            if w.wave_number == self.state.current_wave:
                return i
        return 0

    # -- Internal: wave progression -----------------------------------------

    def _check_wave_completion(self) -> None:
        """Check if current wave is cleared and start next."""
        if self._all_waves_done or not self._wave_started:
            return
        if self._wave_spawn_queue:
            return  # Still spawning

        alive_hostiles = sum(
            1 for u in self._units.values()
            if u.alliance == Alliance.HOSTILE and u.state.is_alive
        )

        if alive_hostiles == 0:
            wave_idx = self._get_wave_index()
            self.emit(SimEvent(
                tick=self.state.tick,
                time=self.state.time,
                event_type="wave_end",
                data={"wave": self.state.current_wave},
            ))
            self.state.score += 500  # Wave clear bonus

            # Start next wave
            next_idx = wave_idx + 1
            if next_idx < len(self.config.waves):
                self._begin_wave(next_idx)
            else:
                self._all_waves_done = True

    # -- Internal: game over ------------------------------------------------

    def _check_game_over(self) -> None:
        """Check if the scenario should end."""
        # All objectives completed = victory
        if self.config.objectives:
            all_done = all(o.completed for o in self.config.objectives)
            if all_done:
                self.state.phase = "game_over"
                self.state.result = "victory"
                self.emit(SimEvent(
                    tick=self.state.tick,
                    time=self.state.time,
                    event_type="scenario_end",
                    data={"result": "victory", "reason": "objectives_complete"},
                ))
                return

        # All friendlies dead = defeat
        alive_friendly = sum(
            1 for u in self._units.values()
            if u.alliance == Alliance.FRIENDLY and u.state.is_alive
        )
        if alive_friendly == 0 and self.state.tick > 0:
            # Only defeat if we actually had friendlies
            had_friendlies = any(
                u.alliance == Alliance.FRIENDLY for u in self._units.values()
            )
            if had_friendlies:
                self.state.phase = "game_over"
                self.state.result = "defeat"
                self.emit(SimEvent(
                    tick=self.state.tick,
                    time=self.state.time,
                    event_type="scenario_end",
                    data={"result": "defeat", "reason": "all_friendlies_dead"},
                ))
                return

    # -- Internal: state sync -----------------------------------------------

    def _sync_units_to_state(self) -> None:
        """Serialize live Unit objects into state.units dict."""
        self.state.units = {}
        for uid, unit in self._units.items():
            self.state.units[uid] = {
                "unit_id": uid,
                "name": unit.name,
                "unit_type": unit.unit_type.value,
                "alliance": unit.alliance.value,
                "position": list(unit.position),
                "health": round(unit.state.health, 1),
                "max_health": unit.stats.max_health,
                "is_alive": unit.state.is_alive,
                "status": unit.state.status,
                "kill_count": unit.state.kill_count,
                "damage_dealt": round(unit.state.damage_dealt, 1),
                "damage_taken": round(unit.state.damage_taken, 1),
                "morale": round(unit.state.morale, 3),
                "suppression": round(unit.state.suppression, 3),
            }


# ---------------------------------------------------------------------------
# Preset scenarios
# ---------------------------------------------------------------------------

PRESET_SCENARIOS: dict[str, ScenarioConfig] = {
    "skirmish": ScenarioConfig(
        name="Skirmish",
        description="3 waves of infantry, defend a point",
        tick_rate=10.0,
        max_ticks=6000,
        waves=[
            WaveConfig(
                wave_number=1,
                spawn_delay=1.0,
                hostiles=[
                    {"template": "infantry", "count": 5,
                     "spawn_pos": (10.0, 100.0), "target_pos": (100.0, 100.0)},
                ],
            ),
            WaveConfig(
                wave_number=2,
                spawn_delay=1.0,
                wave_bonus=0.1,
                hostiles=[
                    {"template": "infantry", "count": 7,
                     "spawn_pos": (10.0, 80.0), "target_pos": (100.0, 100.0)},
                ],
            ),
            WaveConfig(
                wave_number=3,
                spawn_delay=0.8,
                wave_bonus=0.2,
                hostiles=[
                    {"template": "infantry", "count": 5,
                     "spawn_pos": (10.0, 120.0), "target_pos": (100.0, 100.0)},
                    {"template": "heavy", "count": 2,
                     "spawn_pos": (10.0, 100.0), "target_pos": (100.0, 100.0)},
                ],
            ),
        ],
        objectives=[
            Objective(objective_type="defend_point", target_value=300.0),
        ],
        friendly_units=[
            {"template": "infantry", "count": 4, "spawn_pos": (100.0, 100.0)},
            {"template": "sniper", "count": 1, "spawn_pos": (110.0, 110.0)},
        ],
    ),

    "assault": ScenarioConfig(
        name="Assault",
        description="5 escalating waves, eliminate all hostiles",
        tick_rate=10.0,
        max_ticks=12000,
        waves=[
            WaveConfig(
                wave_number=i + 1,
                spawn_delay=max(0.5, 2.0 - i * 0.3),
                wave_bonus=i * 0.1,
                hostiles=[
                    {"template": "infantry", "count": 3 + i * 2,
                     "spawn_pos": (10.0, 100.0), "target_pos": (150.0, 100.0)},
                ] + ([
                    {"template": "heavy", "count": i,
                     "spawn_pos": (10.0, 80.0), "target_pos": (150.0, 100.0)},
                ] if i >= 2 else []),
            )
            for i in range(5)
        ],
        objectives=[
            Objective(objective_type="eliminate_all", target_value=1.0),
        ],
        friendly_units=[
            {"template": "infantry", "count": 5, "spawn_pos": (150.0, 100.0)},
            {"template": "sniper", "count": 2, "spawn_pos": (160.0, 110.0)},
            {"template": "heavy", "count": 1, "spawn_pos": (150.0, 90.0)},
        ],
    ),

    "survival": ScenarioConfig(
        name="Survival",
        description="Endless waves, survive as long as possible",
        tick_rate=10.0,
        max_ticks=18000,  # 30 min max
        waves=[
            WaveConfig(
                wave_number=i + 1,
                spawn_delay=max(0.3, 2.0 - i * 0.15),
                wave_bonus=i * 0.15,
                hostiles=[
                    {"template": "infantry", "count": 4 + i * 3,
                     "spawn_pos": (10.0, 100.0), "target_pos": (100.0, 100.0)},
                ] + ([
                    {"template": "heavy", "count": 1 + i // 2,
                     "spawn_pos": (10.0, 50.0), "target_pos": (100.0, 100.0)},
                ] if i >= 3 else []) + ([
                    {"template": "scout", "count": i,
                     "spawn_pos": (190.0, 100.0), "target_pos": (100.0, 100.0)},
                ] if i >= 5 else []),
            )
            for i in range(20)
        ],
        objectives=[
            Objective(objective_type="survive_time", target_value=600.0),
        ],
        friendly_units=[
            {"template": "infantry", "count": 6, "spawn_pos": (100.0, 100.0)},
            {"template": "sniper", "count": 2, "spawn_pos": (110.0, 110.0)},
            {"template": "heavy", "count": 2, "spawn_pos": (100.0, 90.0)},
        ],
    ),

    "sniper_duel": ScenarioConfig(
        name="Sniper Duel",
        description="1v1 sniper at long range",
        tick_rate=10.0,
        max_ticks=3000,
        waves=[
            WaveConfig(
                wave_number=1,
                spawn_delay=0.0,
                hostiles=[
                    {"template": "sniper", "count": 1,
                     "spawn_pos": (10.0, 100.0), "target_pos": (190.0, 100.0)},
                ],
            ),
        ],
        objectives=[
            Objective(objective_type="eliminate_all", target_value=1.0),
        ],
        friendly_units=[
            {"template": "sniper", "count": 1, "spawn_pos": (190.0, 100.0)},
        ],
        map_size=(200.0, 200.0),
    ),
}

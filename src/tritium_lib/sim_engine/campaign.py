# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Campaign and mission chain system for the Tritium sim engine.

Strings scenarios together with persistent state: veteran units carry over,
XP accumulates, weapons and units unlock, faction reputation shifts, and
resources deplete or grow based on mission performance.

Usage::

    from tritium_lib.sim_engine.campaign import Campaign, CAMPAIGNS

    campaign = Campaign.from_preset("tutorial")
    briefing = campaign.current_mission()
    config = campaign.start_mission()
    # ... run the scenario ...
    campaign.complete_mission(result)
    saved = campaign.save()
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.scenario import (
    ScenarioConfig,
    WaveConfig,
    Objective,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MissionType(Enum):
    """Types of missions available in campaigns."""

    ASSAULT = "assault"
    DEFENSE = "defense"
    ESCORT = "escort"
    RECON = "recon"
    RESCUE = "rescue"
    DEMOLITION = "demolition"
    STEALTH = "stealth"
    PATROL = "patrol"
    AMBUSH = "ambush"
    SIEGE = "siege"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MissionBriefing:
    """Full briefing for a single mission within a campaign."""

    mission_id: str
    name: str
    description: str
    mission_type: MissionType
    difficulty: int  # 1-5 stars
    objectives: list[dict] = field(default_factory=list)
    # Each objective: {"type": "eliminate_all"|"survive_time"|...,
    #                  "target": <value>, "description": "..."}
    available_units: list[str] = field(default_factory=list)  # template names
    max_units: int = 10
    time_limit: float | None = None  # seconds, None = unlimited
    environment: dict = field(default_factory=dict)
    # {"weather": "clear"|"rain"|"fog"|"storm",
    #  "time_of_day": "day"|"night"|"dawn"|"dusk",
    #  "terrain_type": "urban"|"desert"|"forest"|"arctic"|"coastal"}
    enemy_composition: list[dict] = field(default_factory=list)
    # [{"template": "infantry", "count": 5}, ...]
    rewards: dict = field(default_factory=dict)
    # {"xp": 500, "unlock_units": ["sniper"], "unlock_weapons": ["rpg"]}

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "mission_id": self.mission_id,
            "name": self.name,
            "description": self.description,
            "mission_type": self.mission_type.value,
            "difficulty": self.difficulty,
            "objectives": list(self.objectives),
            "available_units": list(self.available_units),
            "max_units": self.max_units,
            "time_limit": self.time_limit,
            "environment": dict(self.environment),
            "enemy_composition": list(self.enemy_composition),
            "rewards": dict(self.rewards),
        }


@dataclass
class MissionResult:
    """Outcome of a completed mission."""

    mission_id: str
    success: bool
    time_taken: float
    casualties_friendly: int = 0
    casualties_enemy: int = 0
    objectives_completed: int = 0
    objectives_total: int = 0
    score: int = 0
    grade: str = "C"  # S/A/B/C/D/F
    mvp: str | None = None
    achievements: list[str] = field(default_factory=list)
    xp_earned: int = 0

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "mission_id": self.mission_id,
            "success": self.success,
            "time_taken": round(self.time_taken, 2),
            "casualties_friendly": self.casualties_friendly,
            "casualties_enemy": self.casualties_enemy,
            "objectives_completed": self.objectives_completed,
            "objectives_total": self.objectives_total,
            "score": self.score,
            "grade": self.grade,
            "mvp": self.mvp,
            "achievements": list(self.achievements),
            "xp_earned": self.xp_earned,
        }


@dataclass
class PersistentState:
    """Persistent state that carries across missions in a campaign."""

    campaign_id: str
    current_mission: int = 0  # index into mission list
    completed_missions: list[MissionResult] = field(default_factory=list)
    total_xp: int = 0
    veteran_units: list[dict] = field(default_factory=list)
    # [{"unit_id": "...", "name": "...", "template": "...", "xp": 100,
    #   "kills": 5, "missions_survived": 2}]
    unlocked_units: list[str] = field(default_factory=list)
    unlocked_weapons: list[str] = field(default_factory=list)
    reputation: dict[str, float] = field(default_factory=dict)
    # faction_id -> standing (-100 to 100)
    resources: dict[str, float] = field(default_factory=dict)
    # ammo_stockpile, fuel_reserve, medical_supplies

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "campaign_id": self.campaign_id,
            "current_mission": self.current_mission,
            "completed_missions": [m.to_dict() for m in self.completed_missions],
            "total_xp": self.total_xp,
            "veteran_units": [dict(v) for v in self.veteran_units],
            "unlocked_units": list(self.unlocked_units),
            "unlocked_weapons": list(self.unlocked_weapons),
            "reputation": dict(self.reputation),
            "resources": dict(self.resources),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PersistentState:
        """Restore from serialized dict."""
        state = cls(campaign_id=data["campaign_id"])
        state.current_mission = data.get("current_mission", 0)
        state.completed_missions = [
            MissionResult(**m) for m in data.get("completed_missions", [])
        ]
        state.total_xp = data.get("total_xp", 0)
        state.veteran_units = data.get("veteran_units", [])
        state.unlocked_units = data.get("unlocked_units", [])
        state.unlocked_weapons = data.get("unlocked_weapons", [])
        state.reputation = data.get("reputation", {})
        state.resources = data.get("resources", {})
        return state


# ---------------------------------------------------------------------------
# Grade calculation
# ---------------------------------------------------------------------------

_GRADE_THRESHOLDS = [
    (95, "S"),
    (85, "A"),
    (70, "B"),
    (50, "C"),
    (30, "D"),
    (0, "F"),
]


def compute_grade(score: int, max_score: int) -> str:
    """Compute letter grade from score as percentage of max."""
    if max_score <= 0:
        return "C"
    pct = (score / max_score) * 100
    for threshold, grade in _GRADE_THRESHOLDS:
        if pct >= threshold:
            return grade
    return "F"


def _grade_to_numeric(grade: str) -> float:
    """Convert letter grade to numeric for averaging."""
    mapping = {"S": 5.0, "A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
    return mapping.get(grade, 2.0)


def _numeric_to_grade(value: float) -> str:
    """Convert numeric average back to letter grade."""
    if value >= 4.5:
        return "S"
    if value >= 3.5:
        return "A"
    if value >= 2.5:
        return "B"
    if value >= 1.5:
        return "C"
    if value >= 0.5:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Campaign class
# ---------------------------------------------------------------------------


class Campaign:
    """A campaign is a sequence of missions with persistent state.

    Manages mission progression, XP rewards, unit veterancy, unlocks,
    and resource tracking across an ordered chain of missions.
    """

    def __init__(
        self,
        campaign_id: str,
        name: str,
        missions: list[MissionBriefing],
        description: str = "",
        initial_resources: dict[str, float] | None = None,
        initial_reputation: dict[str, float] | None = None,
    ) -> None:
        self.campaign_id = campaign_id
        self.name = name
        self.description = description
        self.missions = list(missions)
        self.state = PersistentState(
            campaign_id=campaign_id,
            resources=dict(initial_resources or {
                "ammo_stockpile": 100.0,
                "fuel_reserve": 100.0,
                "medical_supplies": 100.0,
            }),
            reputation=dict(initial_reputation or {}),
        )

    # -- Mission access -----------------------------------------------------

    def current_mission(self) -> MissionBriefing:
        """Return the current mission briefing.

        Raises IndexError if the campaign is complete.
        """
        if self.state.current_mission >= len(self.missions):
            raise IndexError("Campaign is complete, no more missions.")
        return self.missions[self.state.current_mission]

    def start_mission(self) -> dict:
        """Build a ScenarioConfig dict for the current mission.

        Returns a dict that can be used to construct a ScenarioConfig.
        """
        briefing = self.current_mission()

        # Build waves from enemy composition
        waves: list[dict] = []
        wave_count = max(1, len(briefing.enemy_composition))
        for i, enemy_group in enumerate(briefing.enemy_composition):
            wave_dict = {
                "wave_number": i + 1,
                "spawn_delay": max(0.5, 2.0 - i * 0.2),
                "wave_bonus": i * 0.05 * briefing.difficulty,
                "hostiles": [
                    {
                        "template": enemy_group.get("template", "infantry"),
                        "count": enemy_group.get("count", 3),
                        "spawn_pos": enemy_group.get("spawn_pos", [10.0, 100.0]),
                        "target_pos": enemy_group.get("target_pos", [100.0, 100.0]),
                    }
                ],
            }
            waves.append(wave_dict)

        # Build objectives from briefing
        objectives: list[dict] = []
        for obj in briefing.objectives:
            objectives.append({
                "objective_type": obj.get("type", "eliminate_all"),
                "target_value": obj.get("target", 1.0),
            })

        # Build friendly unit config
        friendly_units: list[dict] = []
        for template in briefing.available_units[:briefing.max_units]:
            friendly_units.append({
                "template": template,
                "count": 1,
                "spawn_pos": [100.0, 100.0],
            })

        # Apply veteran bonuses
        veteran_templates = {v["template"]: v for v in self.state.veteran_units}

        # Determine max ticks from time limit
        tick_rate = 10.0
        max_ticks = int(briefing.time_limit * tick_rate) if briefing.time_limit else 6000

        config = {
            "name": briefing.name,
            "description": briefing.description,
            "tick_rate": tick_rate,
            "max_ticks": max_ticks,
            "waves": waves,
            "objectives": objectives,
            "friendly_units": friendly_units,
            "map_size": [200.0, 200.0],
            "mission_id": briefing.mission_id,
            "mission_type": briefing.mission_type.value,
            "difficulty": briefing.difficulty,
            "environment": briefing.environment,
            "veteran_units": list(self.state.veteran_units),
            "unlocked_weapons": list(self.state.unlocked_weapons),
        }

        # Deduct resource costs based on difficulty
        cost_factor = briefing.difficulty * 5.0
        for resource in ["ammo_stockpile", "fuel_reserve"]:
            if resource in self.state.resources:
                self.state.resources[resource] = max(
                    0.0, self.state.resources[resource] - cost_factor
                )

        return config

    def complete_mission(self, result: MissionResult) -> None:
        """Record a mission result and advance the campaign.

        Applies rewards, updates veterancy, handles resource changes.
        """
        briefing = self.current_mission()

        # Record result
        self.state.completed_missions.append(result)

        # Apply XP
        self.state.total_xp += result.xp_earned

        # Apply rewards from briefing
        rewards = briefing.rewards
        if rewards:
            bonus_xp = rewards.get("xp", 0)
            if result.success:
                self.state.total_xp += bonus_xp

            # Unlock units
            for unit_template in rewards.get("unlock_units", []):
                if unit_template not in self.state.unlocked_units:
                    self.state.unlocked_units.append(unit_template)

            # Unlock weapons
            for weapon in rewards.get("unlock_weapons", []):
                if weapon not in self.state.unlocked_weapons:
                    self.state.unlocked_weapons.append(weapon)

        # Veteran units: surviving friendly units gain XP
        if result.success:
            survived_count = max(
                0,
                len(briefing.available_units) - result.casualties_friendly,
            )
            # Add or update veteran units
            for i in range(min(survived_count, len(briefing.available_units))):
                template = briefing.available_units[i]
                vet_id = f"vet_{briefing.mission_id}_{i}"
                # Check if this veteran already exists
                existing = None
                for v in self.state.veteran_units:
                    if v.get("template") == template and v.get("unit_id") == vet_id:
                        existing = v
                        break
                if existing:
                    existing["xp"] = existing.get("xp", 0) + result.xp_earned // max(survived_count, 1)
                    existing["missions_survived"] = existing.get("missions_survived", 0) + 1
                    existing["kills"] = existing.get("kills", 0) + result.casualties_enemy // max(survived_count, 1)
                else:
                    self.state.veteran_units.append({
                        "unit_id": vet_id,
                        "name": f"Veteran {template.title()}",
                        "template": template,
                        "xp": result.xp_earned // max(survived_count, 1),
                        "kills": result.casualties_enemy // max(survived_count, 1),
                        "missions_survived": 1,
                    })

        # Resource recovery on success
        if result.success:
            recovery = 10.0 * (1.0 + result.score / 1000.0)
            for resource in ["medical_supplies"]:
                if resource in self.state.resources:
                    self.state.resources[resource] = min(
                        100.0, self.state.resources[resource] + recovery
                    )

        # Reputation changes
        if result.success:
            for faction_id in self.state.reputation:
                self.state.reputation[faction_id] = min(
                    100.0,
                    self.state.reputation[faction_id] + 5.0,
                )
        else:
            for faction_id in self.state.reputation:
                self.state.reputation[faction_id] = max(
                    -100.0,
                    self.state.reputation[faction_id] - 3.0,
                )

        # Advance to next mission
        self.state.current_mission += 1

    def is_complete(self) -> bool:
        """Return True if all missions have been completed."""
        return self.state.current_mission >= len(self.missions)

    def overall_grade(self) -> str:
        """Compute average grade across all completed missions."""
        if not self.state.completed_missions:
            return "C"
        total = sum(
            _grade_to_numeric(m.grade) for m in self.state.completed_missions
        )
        avg = total / len(self.state.completed_missions)
        return _numeric_to_grade(avg)

    def save(self) -> dict:
        """Serialize full campaign state for persistence."""
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "description": self.description,
            "missions": [m.to_dict() for m in self.missions],
            "state": self.state.to_dict(),
        }

    def load(self, data: dict) -> None:
        """Restore campaign from saved state."""
        self.campaign_id = data.get("campaign_id", self.campaign_id)
        self.name = data.get("name", self.name)
        self.description = data.get("description", self.description)

        if "missions" in data:
            self.missions = []
            for m in data["missions"]:
                mt = m.get("mission_type", "assault")
                if isinstance(mt, str):
                    mt = MissionType(mt)
                self.missions.append(MissionBriefing(
                    mission_id=m["mission_id"],
                    name=m["name"],
                    description=m.get("description", ""),
                    mission_type=mt,
                    difficulty=m.get("difficulty", 1),
                    objectives=m.get("objectives", []),
                    available_units=m.get("available_units", []),
                    max_units=m.get("max_units", 10),
                    time_limit=m.get("time_limit"),
                    environment=m.get("environment", {}),
                    enemy_composition=m.get("enemy_composition", []),
                    rewards=m.get("rewards", {}),
                ))

        if "state" in data:
            self.state = PersistentState.from_dict(data["state"])

    def to_three_js(self) -> dict:
        """Return campaign data formatted for Three.js rendering.

        Includes campaign map overview, mission status indicators,
        and overall progress metrics.
        """
        mission_nodes = []
        for i, mission in enumerate(self.missions):
            status = "locked"
            if i < self.state.current_mission:
                status = "completed"
                # Find the result for grade coloring
                if i < len(self.state.completed_missions):
                    result = self.state.completed_missions[i]
                    if not result.success:
                        status = "failed"
            elif i == self.state.current_mission:
                status = "current"

            # Position missions in a connected chain layout
            angle = (i / max(len(self.missions), 1)) * math.pi * 2
            radius = 80.0
            x = 100.0 + radius * math.cos(angle)
            y = 100.0 + radius * math.sin(angle)

            node: dict[str, Any] = {
                "mission_id": mission.mission_id,
                "name": mission.name,
                "mission_type": mission.mission_type.value,
                "difficulty": mission.difficulty,
                "status": status,
                "position": {"x": round(x, 1), "y": round(y, 1)},
                "index": i,
            }
            if status == "completed" and i < len(self.state.completed_missions):
                node["grade"] = self.state.completed_missions[i].grade
                node["score"] = self.state.completed_missions[i].score
            mission_nodes.append(node)

        # Build connection edges between missions
        edges = []
        for i in range(len(self.missions) - 1):
            edges.append({
                "from": self.missions[i].mission_id,
                "to": self.missions[i + 1].mission_id,
                "unlocked": i < self.state.current_mission,
            })

        progress = (
            self.state.current_mission / len(self.missions)
            if self.missions else 0.0
        )

        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "description": self.description,
            "missions": mission_nodes,
            "edges": edges,
            "progress": round(progress, 3),
            "total_xp": self.state.total_xp,
            "overall_grade": self.overall_grade(),
            "veteran_count": len(self.state.veteran_units),
            "resources": dict(self.state.resources),
            "is_complete": self.is_complete(),
        }

    @classmethod
    def from_preset(cls, preset_name: str) -> Campaign:
        """Create a Campaign from the CAMPAIGNS presets dict.

        Raises KeyError if preset_name is not found.
        """
        if preset_name not in CAMPAIGNS:
            raise KeyError(f"Unknown campaign preset: {preset_name!r}")
        preset = CAMPAIGNS[preset_name]
        return cls(
            campaign_id=preset["campaign_id"],
            name=preset["name"],
            missions=preset["missions"],
            description=preset.get("description", ""),
            initial_resources=preset.get("initial_resources"),
            initial_reputation=preset.get("initial_reputation"),
        )


# ---------------------------------------------------------------------------
# Campaign presets
# ---------------------------------------------------------------------------

CAMPAIGNS: dict[str, dict] = {
    "tutorial": {
        "campaign_id": "tutorial",
        "name": "Basic Training",
        "description": "Three introductory missions teaching core mechanics.",
        "missions": [
            MissionBriefing(
                mission_id="tut_01",
                name="First Contact",
                description="Learn basic movement and engagement. Eliminate a small patrol.",
                mission_type=MissionType.ASSAULT,
                difficulty=1,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Eliminate all hostiles"},
                ],
                available_units=["infantry", "infantry", "infantry"],
                max_units=3,
                time_limit=300.0,
                environment={"weather": "clear", "time_of_day": "day",
                             "terrain_type": "urban"},
                enemy_composition=[
                    {"template": "infantry", "count": 3},
                ],
                rewards={"xp": 100, "unlock_units": [], "unlock_weapons": []},
            ),
            MissionBriefing(
                mission_id="tut_02",
                name="Defensive Posture",
                description="Hold your position against two waves of attackers.",
                mission_type=MissionType.DEFENSE,
                difficulty=1,
                objectives=[
                    {"type": "survive_time", "target": 120.0,
                     "description": "Survive for 2 minutes"},
                ],
                available_units=["infantry", "infantry", "heavy"],
                max_units=3,
                time_limit=180.0,
                environment={"weather": "clear", "time_of_day": "day",
                             "terrain_type": "urban"},
                enemy_composition=[
                    {"template": "infantry", "count": 4},
                    {"template": "infantry", "count": 5},
                ],
                rewards={"xp": 150, "unlock_units": ["sniper"],
                         "unlock_weapons": []},
            ),
            MissionBriefing(
                mission_id="tut_03",
                name="Sniper Introduction",
                description="Use your new sniper to provide overwatch and eliminate targets.",
                mission_type=MissionType.RECON,
                difficulty=2,
                objectives=[
                    {"type": "kill_count", "target": 5.0,
                     "description": "Eliminate 5 hostiles"},
                ],
                available_units=["infantry", "infantry", "sniper"],
                max_units=3,
                environment={"weather": "clear", "time_of_day": "dawn",
                             "terrain_type": "forest"},
                enemy_composition=[
                    {"template": "infantry", "count": 3},
                    {"template": "scout", "count": 3},
                ],
                rewards={"xp": 200, "unlock_units": ["scout"],
                         "unlock_weapons": ["smoke_grenade"]},
            ),
        ],
    },

    "urban_warfare": {
        "campaign_id": "urban_warfare",
        "name": "Urban Warfare",
        "description": "Seven-mission city assault campaign. Clear districts, "
                       "rescue civilians, and hold the city center.",
        "initial_reputation": {"civilians": 50.0, "militia": -20.0},
        "missions": [
            MissionBriefing(
                mission_id="uw_01",
                name="Outskirts Clearing",
                description="Clear the eastern outskirts of hostile patrols.",
                mission_type=MissionType.ASSAULT,
                difficulty=2,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Clear all hostiles"},
                ],
                available_units=["infantry"] * 4 + ["scout"],
                max_units=5,
                time_limit=600.0,
                environment={"weather": "clear", "time_of_day": "dawn",
                             "terrain_type": "urban"},
                enemy_composition=[
                    {"template": "infantry", "count": 6},
                    {"template": "scout", "count": 2},
                ],
                rewards={"xp": 250, "unlock_units": [],
                         "unlock_weapons": ["flashbang"]},
            ),
            MissionBriefing(
                mission_id="uw_02",
                name="Bridge Ambush",
                description="Set up an ambush at the river crossing.",
                mission_type=MissionType.AMBUSH,
                difficulty=2,
                objectives=[
                    {"type": "kill_count", "target": 8.0,
                     "description": "Eliminate 8 hostiles"},
                ],
                available_units=["infantry"] * 3 + ["sniper", "scout"],
                max_units=5,
                environment={"weather": "fog", "time_of_day": "dawn",
                             "terrain_type": "urban"},
                enemy_composition=[
                    {"template": "infantry", "count": 5},
                    {"template": "heavy", "count": 3},
                ],
                rewards={"xp": 300, "unlock_units": ["heavy"],
                         "unlock_weapons": []},
            ),
            MissionBriefing(
                mission_id="uw_03",
                name="Hospital Rescue",
                description="Rescue trapped civilians from the bombed hospital.",
                mission_type=MissionType.RESCUE,
                difficulty=3,
                objectives=[
                    {"type": "survive_time", "target": 180.0,
                     "description": "Secure the hospital for 3 minutes"},
                ],
                available_units=["infantry"] * 3 + ["medic", "heavy"],
                max_units=5,
                environment={"weather": "rain", "time_of_day": "day",
                             "terrain_type": "urban"},
                enemy_composition=[
                    {"template": "infantry", "count": 6},
                    {"template": "infantry", "count": 4},
                ],
                rewards={"xp": 400, "unlock_units": ["medic"],
                         "unlock_weapons": []},
            ),
            MissionBriefing(
                mission_id="uw_04",
                name="Market District",
                description="Clear the market district with minimal collateral damage.",
                mission_type=MissionType.STEALTH,
                difficulty=3,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Eliminate all hostiles"},
                ],
                available_units=["scout"] * 2 + ["sniper", "infantry"],
                max_units=4,
                environment={"weather": "clear", "time_of_day": "night",
                             "terrain_type": "urban"},
                enemy_composition=[
                    {"template": "infantry", "count": 4},
                    {"template": "scout", "count": 4},
                ],
                rewards={"xp": 350, "unlock_units": [],
                         "unlock_weapons": ["silencer"]},
            ),
            MissionBriefing(
                mission_id="uw_05",
                name="Supply Depot Raid",
                description="Raid the enemy supply depot and destroy stockpiles.",
                mission_type=MissionType.DEMOLITION,
                difficulty=3,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Destroy all supply caches"},
                ],
                available_units=["infantry"] * 3 + ["heavy", "engineer"],
                max_units=5,
                environment={"weather": "clear", "time_of_day": "night",
                             "terrain_type": "urban"},
                enemy_composition=[
                    {"template": "infantry", "count": 5},
                    {"template": "heavy", "count": 2},
                    {"template": "scout", "count": 2},
                ],
                rewards={"xp": 400, "unlock_units": ["engineer"],
                         "unlock_weapons": ["c4"]},
            ),
            MissionBriefing(
                mission_id="uw_06",
                name="City Center Push",
                description="Push through to the city center and establish a forward base.",
                mission_type=MissionType.ASSAULT,
                difficulty=4,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Clear the city center"},
                ],
                available_units=["infantry"] * 4 + ["heavy"] * 2 + ["sniper", "medic"],
                max_units=8,
                environment={"weather": "storm", "time_of_day": "dusk",
                             "terrain_type": "urban"},
                enemy_composition=[
                    {"template": "infantry", "count": 8},
                    {"template": "heavy", "count": 3},
                    {"template": "sniper", "count": 2},
                ],
                rewards={"xp": 500, "unlock_units": [],
                         "unlock_weapons": ["rpg"]},
            ),
            MissionBriefing(
                mission_id="uw_07",
                name="Final Stand",
                description="Defend the city center against the enemy counterattack.",
                mission_type=MissionType.DEFENSE,
                difficulty=5,
                objectives=[
                    {"type": "survive_time", "target": 300.0,
                     "description": "Hold for 5 minutes"},
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Eliminate all attackers"},
                ],
                available_units=(
                    ["infantry"] * 5 + ["heavy"] * 2 +
                    ["sniper"] * 2 + ["medic", "engineer"]
                ),
                max_units=11,
                environment={"weather": "storm", "time_of_day": "night",
                             "terrain_type": "urban"},
                enemy_composition=[
                    {"template": "infantry", "count": 10},
                    {"template": "heavy", "count": 5},
                    {"template": "sniper", "count": 3},
                    {"template": "scout", "count": 4},
                ],
                rewards={"xp": 1000, "unlock_units": [],
                         "unlock_weapons": ["minigun"]},
            ),
        ],
    },

    "insurgency": {
        "campaign_id": "insurgency",
        "name": "Counterinsurgency",
        "description": "Ten-mission hearts-and-minds campaign. Balance combat "
                       "effectiveness with civilian relations.",
        "initial_reputation": {"locals": 30.0, "insurgents": -50.0,
                               "government": 60.0},
        "initial_resources": {"ammo_stockpile": 80.0, "fuel_reserve": 90.0,
                              "medical_supplies": 70.0},
        "missions": [
            MissionBriefing(
                mission_id=f"ins_{i+1:02d}",
                name=name,
                description=desc,
                mission_type=mtype,
                difficulty=diff,
                objectives=objs,
                available_units=units,
                max_units=max_u,
                environment=env,
                enemy_composition=enemies,
                rewards=rew,
            )
            for i, (name, desc, mtype, diff, objs, units, max_u, env, enemies, rew) in enumerate([
                (
                    "Village Patrol",
                    "Patrol the outer villages and establish presence.",
                    MissionType.PATROL, 1,
                    [{"type": "survive_time", "target": 120.0,
                      "description": "Complete the patrol route"}],
                    ["infantry"] * 4, 4,
                    {"weather": "clear", "time_of_day": "day", "terrain_type": "desert"},
                    [{"template": "scout", "count": 3}],
                    {"xp": 100, "unlock_units": [], "unlock_weapons": []},
                ),
                (
                    "IED Sweep",
                    "Sweep the main road for improvised explosives.",
                    MissionType.PATROL, 2,
                    [{"type": "survive_time", "target": 180.0,
                      "description": "Clear the road safely"}],
                    ["infantry"] * 3 + ["engineer"], 4,
                    {"weather": "clear", "time_of_day": "day", "terrain_type": "desert"},
                    [{"template": "infantry", "count": 4}],
                    {"xp": 200, "unlock_units": ["engineer"], "unlock_weapons": []},
                ),
                (
                    "Market Security",
                    "Provide security for the weekly market gathering.",
                    MissionType.DEFENSE, 2,
                    [{"type": "survive_time", "target": 240.0,
                      "description": "Keep the market safe"}],
                    ["infantry"] * 3 + ["scout"], 4,
                    {"weather": "clear", "time_of_day": "day", "terrain_type": "urban"},
                    [{"template": "infantry", "count": 5}],
                    {"xp": 250, "unlock_units": [], "unlock_weapons": ["flashbang"]},
                ),
                (
                    "Night Raid",
                    "Raid a suspected weapons cache at night.",
                    MissionType.ASSAULT, 3,
                    [{"type": "eliminate_all", "target": 1.0,
                      "description": "Secure the cache"}],
                    ["infantry"] * 3 + ["sniper", "scout"], 5,
                    {"weather": "clear", "time_of_day": "night", "terrain_type": "urban"},
                    [{"template": "infantry", "count": 6},
                     {"template": "heavy", "count": 2}],
                    {"xp": 350, "unlock_units": ["sniper"], "unlock_weapons": []},
                ),
                (
                    "Ambush Alley",
                    "Your convoy was ambushed. Fight through the kill zone.",
                    MissionType.AMBUSH, 3,
                    [{"type": "survive_time", "target": 120.0,
                      "description": "Survive the ambush"}],
                    ["infantry"] * 4 + ["heavy"], 5,
                    {"weather": "clear", "time_of_day": "dusk", "terrain_type": "desert"},
                    [{"template": "infantry", "count": 8},
                     {"template": "scout", "count": 3}],
                    {"xp": 400, "unlock_units": [], "unlock_weapons": ["smoke_grenade"]},
                ),
                (
                    "Hearts and Minds",
                    "Protect the medical convoy delivering supplies to villages.",
                    MissionType.ESCORT, 3,
                    [{"type": "survive_time", "target": 300.0,
                      "description": "Escort convoy safely"}],
                    ["infantry"] * 3 + ["medic", "scout"], 5,
                    {"weather": "rain", "time_of_day": "day", "terrain_type": "desert"},
                    [{"template": "infantry", "count": 5},
                     {"template": "scout", "count": 3}],
                    {"xp": 450, "unlock_units": ["medic"], "unlock_weapons": []},
                ),
                (
                    "Mountain Recon",
                    "Scout the mountain passes for insurgent supply routes.",
                    MissionType.RECON, 3,
                    [{"type": "kill_count", "target": 4.0,
                      "description": "Identify and neutralize scouts"}],
                    ["scout"] * 2 + ["sniper", "infantry"], 4,
                    {"weather": "fog", "time_of_day": "dawn", "terrain_type": "forest"},
                    [{"template": "scout", "count": 5},
                     {"template": "infantry", "count": 3}],
                    {"xp": 350, "unlock_units": [], "unlock_weapons": ["ghillie_suit"]},
                ),
                (
                    "Compound Assault",
                    "Assault the main insurgent compound.",
                    MissionType.ASSAULT, 4,
                    [{"type": "eliminate_all", "target": 1.0,
                      "description": "Clear the compound"}],
                    ["infantry"] * 4 + ["heavy"] * 2 + ["sniper", "medic"], 8,
                    {"weather": "clear", "time_of_day": "night", "terrain_type": "urban"},
                    [{"template": "infantry", "count": 10},
                     {"template": "heavy", "count": 3},
                     {"template": "sniper", "count": 2}],
                    {"xp": 600, "unlock_units": [], "unlock_weapons": ["rpg"]},
                ),
                (
                    "Bridge Defense",
                    "Hold the bridge against counterattack while reinforcements arrive.",
                    MissionType.DEFENSE, 4,
                    [{"type": "survive_time", "target": 360.0,
                      "description": "Hold for 6 minutes"}],
                    ["infantry"] * 4 + ["heavy"] * 2 + ["engineer", "medic"], 8,
                    {"weather": "storm", "time_of_day": "night", "terrain_type": "urban"},
                    [{"template": "infantry", "count": 12},
                     {"template": "heavy", "count": 4},
                     {"template": "scout", "count": 4}],
                    {"xp": 700, "unlock_units": [], "unlock_weapons": ["claymore"]},
                ),
                (
                    "Final Reckoning",
                    "Assault the insurgent headquarters and end the conflict.",
                    MissionType.SIEGE, 5,
                    [{"type": "eliminate_all", "target": 1.0,
                      "description": "Destroy the HQ"},
                     {"type": "survive_time", "target": 300.0,
                      "description": "Hold for reinforcements"}],
                    (["infantry"] * 5 + ["heavy"] * 3 +
                     ["sniper"] * 2 + ["medic", "engineer"]),
                    12,
                    {"weather": "storm", "time_of_day": "night", "terrain_type": "urban"},
                    [{"template": "infantry", "count": 15},
                     {"template": "heavy", "count": 5},
                     {"template": "sniper", "count": 3},
                     {"template": "scout", "count": 5}],
                    {"xp": 1500, "unlock_units": [], "unlock_weapons": ["minigun"]},
                ),
            ])
        ],
    },

    "naval_campaign": {
        "campaign_id": "naval_campaign",
        "name": "Island Hopping",
        "description": "Five-mission naval campaign. Secure islands across the archipelago.",
        "initial_resources": {"ammo_stockpile": 90.0, "fuel_reserve": 80.0,
                              "medical_supplies": 85.0},
        "missions": [
            MissionBriefing(
                mission_id="nav_01",
                name="Beach Recon",
                description="Recon the beach landing zone for the main assault.",
                mission_type=MissionType.RECON,
                difficulty=2,
                objectives=[
                    {"type": "survive_time", "target": 120.0,
                     "description": "Scout the beach zone"}],
                available_units=["scout"] * 2 + ["infantry"],
                max_units=3,
                environment={"weather": "clear", "time_of_day": "dawn",
                             "terrain_type": "coastal"},
                enemy_composition=[{"template": "infantry", "count": 4}],
                rewards={"xp": 200, "unlock_units": [], "unlock_weapons": []},
            ),
            MissionBriefing(
                mission_id="nav_02",
                name="Beach Assault",
                description="Storm the beach and establish a beachhead.",
                mission_type=MissionType.ASSAULT,
                difficulty=3,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Clear the beach"}],
                available_units=["infantry"] * 4 + ["heavy", "medic"],
                max_units=6,
                environment={"weather": "clear", "time_of_day": "dawn",
                             "terrain_type": "coastal"},
                enemy_composition=[
                    {"template": "infantry", "count": 8},
                    {"template": "heavy", "count": 2}],
                rewards={"xp": 400, "unlock_units": ["heavy"],
                         "unlock_weapons": ["naval_gun"]},
            ),
            MissionBriefing(
                mission_id="nav_03",
                name="Jungle Patrol",
                description="Patrol the interior jungle and locate enemy positions.",
                mission_type=MissionType.PATROL,
                difficulty=3,
                objectives=[
                    {"type": "kill_count", "target": 6.0,
                     "description": "Eliminate 6 hostiles"}],
                available_units=["infantry"] * 3 + ["scout"] * 2,
                max_units=5,
                environment={"weather": "rain", "time_of_day": "day",
                             "terrain_type": "forest"},
                enemy_composition=[
                    {"template": "infantry", "count": 5},
                    {"template": "scout", "count": 4}],
                rewards={"xp": 350, "unlock_units": [],
                         "unlock_weapons": ["machete"]},
            ),
            MissionBriefing(
                mission_id="nav_04",
                name="Airfield Capture",
                description="Capture the enemy airfield for air support operations.",
                mission_type=MissionType.ASSAULT,
                difficulty=4,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Secure the airfield"}],
                available_units=(["infantry"] * 4 + ["heavy"] * 2 +
                                 ["sniper", "engineer"]),
                max_units=8,
                environment={"weather": "clear", "time_of_day": "night",
                             "terrain_type": "coastal"},
                enemy_composition=[
                    {"template": "infantry", "count": 10},
                    {"template": "heavy", "count": 3},
                    {"template": "sniper", "count": 2}],
                rewards={"xp": 600, "unlock_units": ["engineer"],
                         "unlock_weapons": []},
            ),
            MissionBriefing(
                mission_id="nav_05",
                name="Fortress Island",
                description="Assault the heavily fortified main island and end the campaign.",
                mission_type=MissionType.SIEGE,
                difficulty=5,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Destroy the fortress"},
                    {"type": "survive_time", "target": 240.0,
                     "description": "Survive the counterattack"}],
                available_units=(["infantry"] * 5 + ["heavy"] * 3 +
                                 ["sniper"] * 2 + ["medic", "engineer"]),
                max_units=12,
                environment={"weather": "storm", "time_of_day": "dusk",
                             "terrain_type": "coastal"},
                enemy_composition=[
                    {"template": "infantry", "count": 15},
                    {"template": "heavy", "count": 5},
                    {"template": "sniper", "count": 3},
                    {"template": "scout", "count": 3}],
                rewards={"xp": 1200, "unlock_units": [],
                         "unlock_weapons": ["naval_bombardment"]},
            ),
        ],
    },

    "air_superiority": {
        "campaign_id": "air_superiority",
        "name": "Air Superiority",
        "description": "Five-mission campaign to gain control of the skies. "
                       "Ground units provide anti-air support.",
        "initial_resources": {"ammo_stockpile": 100.0, "fuel_reserve": 70.0,
                              "medical_supplies": 80.0},
        "missions": [
            MissionBriefing(
                mission_id="air_01",
                name="Radar Outpost",
                description="Destroy the enemy radar installation.",
                mission_type=MissionType.DEMOLITION,
                difficulty=2,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Destroy the radar"}],
                available_units=["infantry"] * 3 + ["engineer"],
                max_units=4,
                environment={"weather": "clear", "time_of_day": "night",
                             "terrain_type": "desert"},
                enemy_composition=[{"template": "infantry", "count": 5}],
                rewards={"xp": 250, "unlock_units": [],
                         "unlock_weapons": ["stinger"]},
            ),
            MissionBriefing(
                mission_id="air_02",
                name="SAM Battery",
                description="Neutralize the enemy SAM battery to clear the air corridor.",
                mission_type=MissionType.ASSAULT,
                difficulty=3,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Destroy SAM sites"}],
                available_units=["infantry"] * 3 + ["heavy", "engineer"],
                max_units=5,
                environment={"weather": "fog", "time_of_day": "dawn",
                             "terrain_type": "desert"},
                enemy_composition=[
                    {"template": "infantry", "count": 6},
                    {"template": "heavy", "count": 3}],
                rewards={"xp": 400, "unlock_units": ["heavy"],
                         "unlock_weapons": []},
            ),
            MissionBriefing(
                mission_id="air_03",
                name="Forward Airbase",
                description="Capture the forward airbase for allied operations.",
                mission_type=MissionType.ASSAULT,
                difficulty=3,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Secure the airbase"}],
                available_units=["infantry"] * 4 + ["heavy", "sniper"],
                max_units=6,
                environment={"weather": "clear", "time_of_day": "day",
                             "terrain_type": "desert"},
                enemy_composition=[
                    {"template": "infantry", "count": 8},
                    {"template": "heavy", "count": 2},
                    {"template": "scout", "count": 2}],
                rewards={"xp": 450, "unlock_units": ["sniper"],
                         "unlock_weapons": ["javelin"]},
            ),
            MissionBriefing(
                mission_id="air_04",
                name="Convoy Escort",
                description="Escort the AA battery convoy to the forward position.",
                mission_type=MissionType.ESCORT,
                difficulty=4,
                objectives=[
                    {"type": "survive_time", "target": 300.0,
                     "description": "Protect the convoy"}],
                available_units=["infantry"] * 4 + ["heavy"] * 2 + ["medic", "scout"],
                max_units=8,
                environment={"weather": "clear", "time_of_day": "dusk",
                             "terrain_type": "desert"},
                enemy_composition=[
                    {"template": "infantry", "count": 10},
                    {"template": "heavy", "count": 3},
                    {"template": "scout", "count": 4}],
                rewards={"xp": 600, "unlock_units": ["medic"],
                         "unlock_weapons": []},
            ),
            MissionBriefing(
                mission_id="air_05",
                name="Sky Hammer",
                description="Final assault on the enemy air command center. "
                            "Achieve total air dominance.",
                mission_type=MissionType.SIEGE,
                difficulty=5,
                objectives=[
                    {"type": "eliminate_all", "target": 1.0,
                     "description": "Destroy the command center"},
                    {"type": "survive_time", "target": 240.0,
                     "description": "Hold the perimeter"}],
                available_units=(["infantry"] * 5 + ["heavy"] * 3 +
                                 ["sniper"] * 2 + ["medic", "engineer"]),
                max_units=12,
                environment={"weather": "storm", "time_of_day": "night",
                             "terrain_type": "desert"},
                enemy_composition=[
                    {"template": "infantry", "count": 12},
                    {"template": "heavy", "count": 5},
                    {"template": "sniper", "count": 3},
                    {"template": "scout", "count": 4}],
                rewards={"xp": 1500, "unlock_units": [],
                         "unlock_weapons": ["airstrike"]},
            ),
        ],
    },
}

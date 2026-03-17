"""Scoring, achievements, statistics, and after-action review for the Tritium sim engine.

Tracks per-unit and per-team combat statistics, unlocks achievements based on
performance thresholds, builds an event timeline, and generates structured
after-action reports suitable for Three.js overlay rendering.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ScoreCategory(enum.Enum):
    """Broad scoring dimensions."""

    KILLS = "kills"
    ASSISTS = "assists"
    OBJECTIVES = "objectives"
    SURVIVAL = "survival"
    ACCURACY = "accuracy"
    TEAMWORK = "teamwork"
    TACTICAL = "tactical"
    ECONOMY = "economy"


# ---------------------------------------------------------------------------
# Achievement definition
# ---------------------------------------------------------------------------


@dataclass
class Achievement:
    """A named milestone that awards bonus score when unlocked."""

    achievement_id: str
    name: str
    description: str
    category: ScoreCategory
    threshold: float
    points: int
    icon: str

    # Optional: callable key used internally for dynamic checks.
    # Not serialized.  Set by ScoringEngine when registering custom checks.
    _check_key: str = ""


# ---------------------------------------------------------------------------
# Scorecards
# ---------------------------------------------------------------------------


@dataclass
class UnitScorecard:
    """Per-unit running statistics."""

    unit_id: str
    name: str
    alliance: str

    kills: int = 0
    deaths: int = 0
    assists: int = 0
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    healing_done: float = 0.0
    shots_fired: int = 0
    shots_hit: int = 0
    objectives_completed: int = 0
    time_alive: float = 0.0
    distance_moved: float = 0.0
    time_in_cover: float = 0.0
    allies_revived: int = 0
    vehicles_destroyed: int = 0
    structures_destroyed: int = 0
    supplies_used: dict[str, float] = field(default_factory=dict)
    achievements: list[str] = field(default_factory=list)

    # Internal tracking
    _last_pos: Optional[Vec2] = field(default=None, repr=False)
    _kill_times: list[float] = field(default_factory=list, repr=False)
    _low_health_time: float = field(default=0.0, repr=False)
    _detected: bool = field(default=False, repr=False)

    # ---- computed properties ----

    @property
    def kd_ratio(self) -> float:
        """Kill/death ratio.  Returns kills if zero deaths."""
        if self.deaths == 0:
            return float(self.kills)
        return self.kills / self.deaths

    @property
    def accuracy(self) -> float:
        """Shot accuracy as a fraction 0-1."""
        if self.shots_fired == 0:
            return 0.0
        return self.shots_hit / self.shots_fired

    @property
    def score(self) -> int:
        """Weighted composite score."""
        s = 0
        s += self.kills * 100
        s += self.assists * 50
        s += self.objectives_completed * 200
        s += self.vehicles_destroyed * 150
        s += self.structures_destroyed * 100
        s += self.allies_revived * 75
        s += int(self.healing_done * 0.5)
        s += int(self.damage_dealt * 0.25)
        # Accuracy bonus
        if self.shots_fired >= 10:
            s += int(self.accuracy * 200)
        # Survival bonus
        s += int(self.time_alive * 0.5)
        return s

    def to_dict(self) -> dict:
        """Serialize to plain dict (omits private fields)."""
        return {
            "unit_id": self.unit_id,
            "name": self.name,
            "alliance": self.alliance,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "damage_dealt": round(self.damage_dealt, 1),
            "damage_taken": round(self.damage_taken, 1),
            "healing_done": round(self.healing_done, 1),
            "shots_fired": self.shots_fired,
            "shots_hit": self.shots_hit,
            "accuracy": round(self.accuracy, 3),
            "kd_ratio": round(self.kd_ratio, 2),
            "objectives_completed": self.objectives_completed,
            "time_alive": round(self.time_alive, 1),
            "distance_moved": round(self.distance_moved, 1),
            "time_in_cover": round(self.time_in_cover, 1),
            "allies_revived": self.allies_revived,
            "vehicles_destroyed": self.vehicles_destroyed,
            "structures_destroyed": self.structures_destroyed,
            "supplies_used": self.supplies_used,
            "achievements": list(self.achievements),
            "score": self.score,
        }


@dataclass
class TeamScorecard:
    """Aggregate stats for one alliance."""

    alliance: str
    total_kills: int = 0
    total_deaths: int = 0
    objectives_completed: int = 0
    objectives_total: int = 0
    territory_controlled: float = 0.0  # 0-1
    casualties_pct: float = 0.0
    supplies_remaining: float = 1.0
    unit_scores: list[UnitScorecard] = field(default_factory=list)

    @property
    def mvp(self) -> Optional[UnitScorecard]:
        """Highest-scoring unit on this team."""
        if not self.unit_scores:
            return None
        return max(self.unit_scores, key=lambda u: u.score)

    @property
    def is_victorious(self) -> bool:
        """Simple heuristic: more objectives completed or more kills if tied."""
        # This is meant to be set externally or compared across teams.
        # As a standalone check, treat >50% territory as a win indicator.
        return self.territory_controlled > 0.5

    def to_dict(self) -> dict:
        mvp_card = self.mvp
        return {
            "alliance": self.alliance,
            "total_kills": self.total_kills,
            "total_deaths": self.total_deaths,
            "objectives_completed": self.objectives_completed,
            "objectives_total": self.objectives_total,
            "territory_controlled": round(self.territory_controlled, 3),
            "casualties_pct": round(self.casualties_pct, 3),
            "supplies_remaining": round(self.supplies_remaining, 3),
            "mvp": mvp_card.name if mvp_card else None,
            "mvp_score": mvp_card.score if mvp_card else 0,
            "unit_count": len(self.unit_scores),
        }


# ---------------------------------------------------------------------------
# Default achievements (20+)
# ---------------------------------------------------------------------------

ACHIEVEMENTS: list[Achievement] = [
    Achievement("first_blood", "First Blood", "Score the first kill of the match",
                ScoreCategory.KILLS, 1, 100, "\u2694\ufe0f"),
    Achievement("sharpshooter", "Sharpshooter", "Maintain 80%+ accuracy (min 20 shots)",
                ScoreCategory.ACCURACY, 0.80, 200, "\ud83c\udfaf"),
    Achievement("rampage", "Rampage", "5 kills within 30 seconds",
                ScoreCategory.KILLS, 5, 300, "\ud83d\udd25"),
    Achievement("medic", "Medic!", "Heal 500 HP total",
                ScoreCategory.TEAMWORK, 500, 150, "\u2695\ufe0f"),
    Achievement("untouchable", "Untouchable", "Take zero damage the entire match",
                ScoreCategory.SURVIVAL, 0, 250, "\ud83d\udee1\ufe0f"),
    Achievement("architect", "Architect", "Destroy 3 structures",
                ScoreCategory.TACTICAL, 3, 200, "\ud83c\udfd7\ufe0f"),
    Achievement("convoy_killer", "Convoy Killer", "Destroy 3 vehicles",
                ScoreCategory.KILLS, 3, 250, "\ud83d\udca5"),
    Achievement("ace", "Ace", "Shoot down 3 aircraft",
                ScoreCategory.KILLS, 3, 300, "\u2708\ufe0f"),
    Achievement("lone_wolf", "Lone Wolf", "Get 10 kills without squad assists",
                ScoreCategory.KILLS, 10, 200, "\ud83d\udc3a"),
    Achievement("team_player", "Team Player", "Earn 5 assists",
                ScoreCategory.ASSISTS, 5, 150, "\ud83e\udd1d"),
    Achievement("demolitions_expert", "Demolitions Expert", "Deal 500 explosive damage",
                ScoreCategory.TACTICAL, 500, 200, "\ud83d\udca3"),
    Achievement("ghost", "Ghost", "Complete the mission undetected",
                ScoreCategory.TACTICAL, 0, 400, "\ud83d\udc7b"),
    Achievement("iron_will", "Iron Will", "Survive below 10% health for 30 seconds",
                ScoreCategory.SURVIVAL, 30, 250, "\ud83e\uddbe"),
    Achievement("ammo_conservation", "Ammo Conservation", "Win with >50% ammo remaining",
                ScoreCategory.ECONOMY, 0.5, 150, "\ud83d\udce6"),
    Achievement("flawless_victory", "Flawless Victory", "Team wins with zero deaths",
                ScoreCategory.TEAMWORK, 0, 500, "\ud83c\udfc6"),
    Achievement("double_kill", "Double Kill", "2 kills within 5 seconds",
                ScoreCategory.KILLS, 2, 100, "\u26a1"),
    Achievement("triple_kill", "Triple Kill", "3 kills within 10 seconds",
                ScoreCategory.KILLS, 3, 150, "\u26a1\u26a1"),
    Achievement("objective_master", "Objective Master", "Complete 3 objectives",
                ScoreCategory.OBJECTIVES, 3, 200, "\ud83c\udf1f"),
    Achievement("field_surgeon", "Field Surgeon", "Revive 3 allies",
                ScoreCategory.TEAMWORK, 3, 200, "\ud83c\udfe5"),
    Achievement("marathon", "Marathon Runner", "Move 5000 meters total",
                ScoreCategory.SURVIVAL, 5000, 100, "\ud83c\udfc3"),
    Achievement("centurion", "Centurion", "Score 100 kills",
                ScoreCategory.KILLS, 100, 500, "\ud83d\udc51"),
    Achievement("pacifist", "Pacifist", "Win with zero kills",
                ScoreCategory.TACTICAL, 0, 300, "\u262e\ufe0f"),
    Achievement("tank_buster", "Tank Buster", "Destroy 5 vehicles",
                ScoreCategory.KILLS, 5, 350, "\ud83d\ude80"),
    Achievement("support_mvp", "Support MVP", "Heal 1000 HP and revive 5 allies",
                ScoreCategory.TEAMWORK, 1000, 400, "\u2764\ufe0f"),
]


# ---------------------------------------------------------------------------
# ScoringEngine
# ---------------------------------------------------------------------------


class ScoringEngine:
    """Central scoring, achievement, and after-action review engine.

    Call ``register_unit`` for every combatant before the sim starts, then
    call the ``record_*`` methods as events occur.  ``tick()`` should be
    called every sim frame to update time-based stats.  At the end of the
    match, ``generate_aar()`` produces the full after-action report.
    """

    def __init__(self, achievements: list[Achievement] | None = None) -> None:
        self.unit_scores: dict[str, UnitScorecard] = {}
        self.team_scores: dict[str, TeamScorecard] = {}
        self.achievements: list[Achievement] = list(achievements or ACHIEVEMENTS)
        self.timeline: list[dict] = []
        self._sim_time: float = 0.0
        self._first_kill_awarded: bool = False
        self._kill_positions: list[tuple[float, float, float]] = []  # x, y, t
        self._death_positions: list[tuple[float, float, float]] = []
        self._movement_positions: list[tuple[float, float, float]] = []

    # ---- registration ----

    def register_unit(self, unit_id: str, name: str, alliance: str) -> UnitScorecard:
        """Register a unit for scoring.  Creates team scorecard if needed."""
        card = UnitScorecard(unit_id=unit_id, name=name, alliance=alliance)
        self.unit_scores[unit_id] = card
        if alliance not in self.team_scores:
            self.team_scores[alliance] = TeamScorecard(alliance=alliance)
        self.team_scores[alliance].unit_scores.append(card)
        return card

    # ---- event recording ----

    def record_kill(self, killer_id: str, victim_id: str) -> None:
        """Record a kill event."""
        killer = self.unit_scores.get(killer_id)
        victim = self.unit_scores.get(victim_id)
        if killer:
            killer.kills += 1
            killer._kill_times.append(self._sim_time)
            if killer._last_pos:
                self._kill_positions.append(
                    (killer._last_pos[0], killer._last_pos[1], self._sim_time)
                )
            # Update team
            team = self.team_scores.get(killer.alliance)
            if team:
                team.total_kills += 1
        if victim:
            victim.deaths += 1
            if victim._last_pos:
                self._death_positions.append(
                    (victim._last_pos[0], victim._last_pos[1], self._sim_time)
                )
            team = self.team_scores.get(victim.alliance)
            if team:
                team.total_deaths += 1

        # First blood
        event_data: dict = {
            "t": round(self._sim_time, 2),
            "event": "kill",
            "killer": killer_id,
            "victim": victim_id,
        }
        if not self._first_kill_awarded and killer:
            event_data["first_blood"] = True
            self._first_kill_awarded = True
            self.timeline.append({
                "t": round(self._sim_time, 2),
                "event": "first_blood",
                "unit": killer_id,
            })
        self.timeline.append(event_data)

        # Check multi-kill achievements for killer
        if killer:
            self.check_achievements(killer_id)

    def record_damage(self, source_id: str, target_id: str, amount: float,
                      damage_type: str = "kinetic") -> None:
        """Record damage dealt from source to target."""
        src = self.unit_scores.get(source_id)
        tgt = self.unit_scores.get(target_id)
        if src:
            src.damage_dealt += amount
        if tgt:
            tgt.damage_taken += amount

    def record_shot(self, shooter_id: str, hit: bool) -> None:
        """Record a shot fired (hit or miss)."""
        card = self.unit_scores.get(shooter_id)
        if card:
            card.shots_fired += 1
            if hit:
                card.shots_hit += 1

    def record_healing(self, healer_id: str, patient_id: str, amount: float) -> None:
        """Record healing performed."""
        healer = self.unit_scores.get(healer_id)
        if healer:
            healer.healing_done += amount
        self.timeline.append({
            "t": round(self._sim_time, 2),
            "event": "healing",
            "healer": healer_id,
            "patient": patient_id,
            "amount": round(amount, 1),
        })

    def record_objective(self, unit_id: str, objective_name: str) -> None:
        """Record an objective completion."""
        card = self.unit_scores.get(unit_id)
        if card:
            card.objectives_completed += 1
            team = self.team_scores.get(card.alliance)
            if team:
                team.objectives_completed += 1
        self.timeline.append({
            "t": round(self._sim_time, 2),
            "event": "objective",
            "unit": unit_id,
            "objective": objective_name,
        })

    def record_revive(self, medic_id: str, patient_id: str) -> None:
        """Record an ally revive."""
        card = self.unit_scores.get(medic_id)
        if card:
            card.allies_revived += 1
        self.timeline.append({
            "t": round(self._sim_time, 2),
            "event": "revive",
            "medic": medic_id,
            "patient": patient_id,
        })

    def record_vehicle_destroyed(self, unit_id: str) -> None:
        """Record a vehicle destruction by unit."""
        card = self.unit_scores.get(unit_id)
        if card:
            card.vehicles_destroyed += 1

    def record_structure_destroyed(self, unit_id: str) -> None:
        """Record a structure destruction by unit."""
        card = self.unit_scores.get(unit_id)
        if card:
            card.structures_destroyed += 1

    def record_event(self, event_type: str, data: dict | None = None) -> None:
        """Record a generic timeline event."""
        entry: dict = {
            "t": round(self._sim_time, 2),
            "event": event_type,
        }
        if data:
            entry.update(data)
        self.timeline.append(entry)

    def record_supply_use(self, unit_id: str, supply_type: str, amount: float) -> None:
        """Record supply consumption."""
        card = self.unit_scores.get(unit_id)
        if card:
            card.supplies_used[supply_type] = card.supplies_used.get(supply_type, 0.0) + amount

    def record_detection(self, unit_id: str) -> None:
        """Mark a unit as having been detected (breaks Ghost achievement)."""
        card = self.unit_scores.get(unit_id)
        if card:
            card._detected = True

    # ---- tick (called every frame) ----

    def tick(self, dt: float, alive_units: set[str] | None = None,
             unit_positions: dict[str, Vec2] | None = None) -> None:
        """Advance time-based statistics.

        Args:
            dt: delta time in seconds since last tick.
            alive_units: set of unit IDs currently alive.
            unit_positions: current position of each alive unit.
        """
        self._sim_time += dt

        alive = alive_units or set(self.unit_scores.keys())
        positions = unit_positions or {}

        for uid, card in self.unit_scores.items():
            if uid in alive:
                card.time_alive += dt

            pos = positions.get(uid)
            if pos is not None and card._last_pos is not None:
                d = distance(card._last_pos, pos)
                if d > 0.01:  # ignore micro-jitter
                    card.distance_moved += d
                    self._movement_positions.append(
                        (pos[0], pos[1], self._sim_time)
                    )
            if pos is not None:
                card._last_pos = pos

    # ---- achievement checking ----

    def check_achievements(self, unit_id: str) -> list[Achievement]:
        """Check and award any newly-earned achievements for a unit."""
        card = self.unit_scores.get(unit_id)
        if card is None:
            return []

        newly_earned: list[Achievement] = []

        for ach in self.achievements:
            if ach.achievement_id in card.achievements:
                continue  # already earned

            earned = False

            if ach.achievement_id == "first_blood":
                # Awarded inline by record_kill, but check here too
                earned = self._first_kill_awarded and card.kills >= 1 and \
                    any(e.get("first_blood") and e.get("killer") == unit_id
                        for e in self.timeline)

            elif ach.achievement_id == "sharpshooter":
                earned = card.shots_fired >= 20 and card.accuracy >= ach.threshold

            elif ach.achievement_id == "rampage":
                earned = self._check_multi_kill(card, count=5, window=30.0)

            elif ach.achievement_id == "medic":
                earned = card.healing_done >= ach.threshold

            elif ach.achievement_id == "untouchable":
                # Only awarded at end of match via generate_aar, but allow
                # early check: if damage_taken > 0, definitely not earned.
                earned = False  # deferred to AAR

            elif ach.achievement_id == "architect":
                earned = card.structures_destroyed >= ach.threshold

            elif ach.achievement_id == "convoy_killer":
                earned = card.vehicles_destroyed >= ach.threshold

            elif ach.achievement_id == "ace":
                # Would need aircraft tracking; use vehicles_destroyed as proxy
                earned = False  # deferred — requires flight-specific tracking

            elif ach.achievement_id == "lone_wolf":
                earned = card.kills >= ach.threshold and card.assists == 0

            elif ach.achievement_id == "team_player":
                earned = card.assists >= ach.threshold

            elif ach.achievement_id == "demolitions_expert":
                # Would need explosive-specific damage tracking
                earned = False  # deferred

            elif ach.achievement_id == "ghost":
                earned = not card._detected and card.kills >= 1

            elif ach.achievement_id == "iron_will":
                earned = card._low_health_time >= ach.threshold

            elif ach.achievement_id == "ammo_conservation":
                earned = False  # deferred to AAR

            elif ach.achievement_id == "flawless_victory":
                earned = False  # deferred to AAR (needs team context)

            elif ach.achievement_id == "double_kill":
                earned = self._check_multi_kill(card, count=2, window=5.0)

            elif ach.achievement_id == "triple_kill":
                earned = self._check_multi_kill(card, count=3, window=10.0)

            elif ach.achievement_id == "objective_master":
                earned = card.objectives_completed >= ach.threshold

            elif ach.achievement_id == "field_surgeon":
                earned = card.allies_revived >= ach.threshold

            elif ach.achievement_id == "marathon":
                earned = card.distance_moved >= ach.threshold

            elif ach.achievement_id == "centurion":
                earned = card.kills >= ach.threshold

            elif ach.achievement_id == "pacifist":
                earned = False  # deferred to AAR

            elif ach.achievement_id == "tank_buster":
                earned = card.vehicles_destroyed >= ach.threshold

            elif ach.achievement_id == "support_mvp":
                earned = card.healing_done >= ach.threshold and card.allies_revived >= 5

            if earned:
                card.achievements.append(ach.achievement_id)
                newly_earned.append(ach)
                self.timeline.append({
                    "t": round(self._sim_time, 2),
                    "event": "achievement",
                    "unit": unit_id,
                    "achievement": ach.achievement_id,
                    "name": ach.name,
                    "points": ach.points,
                })

        return newly_earned

    def _check_multi_kill(self, card: UnitScorecard, count: int, window: float) -> bool:
        """Check if *count* kills occurred within *window* seconds."""
        times = card._kill_times
        if len(times) < count:
            return False
        # Sliding window over sorted kill times
        for i in range(len(times) - count + 1):
            if times[i + count - 1] - times[i] <= window:
                return True
        return False

    # ---- end-of-match deferred achievement awards ----

    def _award_deferred_achievements(self, winner_alliance: str | None) -> None:
        """Award achievements that can only be checked at end of match."""
        for uid, card in self.unit_scores.items():
            # Untouchable
            if card.damage_taken == 0.0 and card.time_alive > 0:
                ach = self._find_achievement("untouchable")
                if ach and ach.achievement_id not in card.achievements:
                    card.achievements.append(ach.achievement_id)
                    self.timeline.append({
                        "t": round(self._sim_time, 2),
                        "event": "achievement",
                        "unit": uid,
                        "achievement": ach.achievement_id,
                        "name": ach.name,
                        "points": ach.points,
                    })

            # Pacifist — win with zero kills
            if winner_alliance and card.alliance == winner_alliance and card.kills == 0:
                ach = self._find_achievement("pacifist")
                if ach and ach.achievement_id not in card.achievements:
                    card.achievements.append(ach.achievement_id)

            # Flawless Victory — team wins with zero deaths
            if winner_alliance and card.alliance == winner_alliance:
                team = self.team_scores.get(card.alliance)
                if team and team.total_deaths == 0:
                    ach = self._find_achievement("flawless_victory")
                    if ach and ach.achievement_id not in card.achievements:
                        card.achievements.append(ach.achievement_id)

    def _find_achievement(self, aid: str) -> Achievement | None:
        for a in self.achievements:
            if a.achievement_id == aid:
                return a
        return None

    # ---- leaderboard ----

    def get_leaderboard(self, category: ScoreCategory | None = None) -> list[dict]:
        """Return ranked unit list, optionally filtered by category."""
        cards = list(self.unit_scores.values())

        if category is None:
            cards.sort(key=lambda c: c.score, reverse=True)
        elif category == ScoreCategory.KILLS:
            cards.sort(key=lambda c: c.kills, reverse=True)
        elif category == ScoreCategory.ASSISTS:
            cards.sort(key=lambda c: c.assists, reverse=True)
        elif category == ScoreCategory.OBJECTIVES:
            cards.sort(key=lambda c: c.objectives_completed, reverse=True)
        elif category == ScoreCategory.SURVIVAL:
            cards.sort(key=lambda c: c.time_alive, reverse=True)
        elif category == ScoreCategory.ACCURACY:
            cards.sort(key=lambda c: c.accuracy, reverse=True)
        elif category == ScoreCategory.TEAMWORK:
            cards.sort(key=lambda c: c.healing_done + c.allies_revived * 100, reverse=True)
        elif category == ScoreCategory.TACTICAL:
            cards.sort(key=lambda c: c.structures_destroyed + c.vehicles_destroyed, reverse=True)
        elif category == ScoreCategory.ECONOMY:
            cards.sort(key=lambda c: -sum(c.supplies_used.values()) if c.supplies_used else 0)

        return [c.to_dict() for c in cards]

    # ---- after-action report ----

    def generate_aar(self, winner_alliance: str | None = None) -> dict:
        """Generate the full after-action report.

        If *winner_alliance* is provided, deferred achievements (Flawless
        Victory, Pacifist, etc.) are awarded before the report is built.

        Returns a dict suitable for JSON serialization.
        """
        self._award_deferred_achievements(winner_alliance)

        # Final achievement sweep for all units
        for uid in list(self.unit_scores):
            self.check_achievements(uid)

        # Compute total stats
        total_kills = sum(t.total_kills for t in self.team_scores.values())
        total_deaths = sum(t.total_deaths for t in self.team_scores.values())

        # Build team summaries
        teams = []
        for alliance, team in self.team_scores.items():
            mvp_card = team.mvp
            teams.append({
                "alliance": alliance,
                "kills": team.total_kills,
                "deaths": team.total_deaths,
                "objectives_completed": team.objectives_completed,
                "objectives_total": team.objectives_total,
                "territory_controlled": round(team.territory_controlled, 3),
                "mvp": mvp_card.name if mvp_card else None,
                "mvp_score": mvp_card.score if mvp_card else 0,
            })

        # Build leaderboard
        leaderboard = self.get_leaderboard()

        # Build achievement list
        achievement_entries = []
        for uid, card in self.unit_scores.items():
            for aid in card.achievements:
                ach = self._find_achievement(aid)
                if ach:
                    achievement_entries.append({
                        "unit": uid,
                        "unit_name": card.name,
                        "achievement": ach.achievement_id,
                        "name": ach.name,
                        "points": ach.points,
                        "icon": ach.icon,
                    })

        return {
            "summary": {
                "duration": round(self._sim_time, 1),
                "winner": winner_alliance,
                "total_kills": total_kills,
                "total_deaths": total_deaths,
                "units_registered": len(self.unit_scores),
            },
            "timeline": self.timeline,
            "teams": teams,
            "leaderboard": leaderboard,
            "achievements": achievement_entries,
            "heatmaps": {
                "kills": [{"x": p[0], "y": p[1], "t": p[2]} for p in self._kill_positions],
                "deaths": [{"x": p[0], "y": p[1], "t": p[2]} for p in self._death_positions],
                "movement": [{"x": p[0], "y": p[1], "t": p[2]} for p in self._movement_positions],
            },
        }

    # ---- Three.js overlay data ----

    def to_three_js(self) -> dict:
        """Return live scoreboard data for Three.js overlay rendering.

        Includes:
        - ``scoreboard``: sorted unit scores
        - ``kill_feed``: recent kill events (last 10)
        - ``achievement_popups``: recent achievements (last 5)
        - ``team_summary``: per-team totals
        """
        # Kill feed — last 10 kill events
        kill_events = [e for e in self.timeline if e.get("event") == "kill"]
        kill_feed = []
        for ev in kill_events[-10:]:
            killer_card = self.unit_scores.get(ev.get("killer", ""))
            victim_card = self.unit_scores.get(ev.get("victim", ""))
            kill_feed.append({
                "t": ev["t"],
                "killer": killer_card.name if killer_card else ev.get("killer"),
                "killer_alliance": killer_card.alliance if killer_card else "unknown",
                "victim": victim_card.name if victim_card else ev.get("victim"),
                "victim_alliance": victim_card.alliance if victim_card else "unknown",
                "first_blood": ev.get("first_blood", False),
            })

        # Achievement popups — last 5
        ach_events = [e for e in self.timeline if e.get("event") == "achievement"]
        popups = []
        for ev in ach_events[-5:]:
            ach = self._find_achievement(ev.get("achievement", ""))
            card = self.unit_scores.get(ev.get("unit", ""))
            popups.append({
                "t": ev["t"],
                "unit": card.name if card else ev.get("unit"),
                "achievement": ev.get("name", ""),
                "icon": ach.icon if ach else "",
                "points": ev.get("points", 0),
            })

        # Team summaries
        team_summary = {
            alliance: team.to_dict()
            for alliance, team in self.team_scores.items()
        }

        return {
            "sim_time": round(self._sim_time, 2),
            "scoreboard": self.get_leaderboard(),
            "kill_feed": kill_feed,
            "achievement_popups": popups,
            "team_summary": team_summary,
        }

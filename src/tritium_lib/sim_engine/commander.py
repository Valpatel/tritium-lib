# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""AI commander, battle narrator, and tactical advisor for the Tritium sim engine.

Generates human-readable battle narration and tactical recommendations from
sim state — like a military radio operator calling out events in real time.

Classes:
    NarrationEvent   — single narration entry with tick, priority, voice
    CommanderPersonality — named personality controlling style/verbosity
    BattleNarrator   — generates varied military-style radio chatter
    TacticalAdvisor  — situation assessment and tactical recommendations
    NarrationLog     — accumulates events, filters, renders to HUD format

Usage::

    from tritium_lib.sim_engine.commander import (
        BattleNarrator, TacticalAdvisor, NarrationLog,
        CommanderPersonality, PERSONALITIES,
    )

    narrator = BattleNarrator(personality=PERSONALITIES["mad_dog"])
    log = NarrationLog()
    event = narrator.narrate_kill(killer_unit, victim_unit)
    log.add(event)
    hud = log.to_three_js()

Copyright 2026 Valpatel Software LLC — AGPL-3.0
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from tritium_lib.sim_engine.ai.steering import Vec2, distance


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

CATEGORIES = ("combat", "tactical", "intel", "medical", "logistics")
VOICES = ("radio", "commander", "observer", "medic")
PRIORITIES = {1: "routine", 2: "important", 3: "urgent", 4: "critical"}


@dataclass
class NarrationEvent:
    """A single narration entry in the battle log."""

    tick: int
    time: float
    category: str  # combat, tactical, intel, medical, logistics
    priority: int  # 1=routine, 2=important, 3=urgent, 4=critical
    text: str
    voice: str = "radio"  # radio, commander, observer, medic

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"Invalid category '{self.category}', must be one of {CATEGORIES}")
        if self.priority not in PRIORITIES:
            raise ValueError(f"Invalid priority {self.priority}, must be 1-4")
        if self.voice not in VOICES:
            raise ValueError(f"Invalid voice '{self.voice}', must be one of {VOICES}")


@dataclass
class CommanderPersonality:
    """Personality profile controlling narration style."""

    name: str
    callsign: str
    style: str  # professional, aggressive, cautious, dramatic
    verbosity: float = 0.5  # 0-1, how much narration
    humor: float = 0.0  # 0-1

    def __post_init__(self) -> None:
        self.verbosity = max(0.0, min(1.0, self.verbosity))
        self.humor = max(0.0, min(1.0, self.humor))


# Preset personalities
PERSONALITIES: dict[str, CommanderPersonality] = {
    "iron_hand": CommanderPersonality(
        name="Iron Hand",
        callsign="IRON",
        style="professional",
        verbosity=0.3,
        humor=0.0,
    ),
    "mad_dog": CommanderPersonality(
        name="Mad Dog",
        callsign="MAD DOG",
        style="aggressive",
        verbosity=0.9,
        humor=0.3,
    ),
    "ghost": CommanderPersonality(
        name="Ghost",
        callsign="GHOST",
        style="cautious",
        verbosity=0.15,
        humor=0.0,
    ),
    "showman": CommanderPersonality(
        name="The Showman",
        callsign="SHOWTIME",
        style="dramatic",
        verbosity=1.0,
        humor=0.6,
    ),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _grid_ref(pos: Vec2) -> str:
    """Convert a Vec2 position to a military-style grid reference."""
    return f"{int(abs(pos[0])):03d}-{int(abs(pos[1])):03d}"


def _bearing_word(from_pos: Vec2, to_pos: Vec2) -> str:
    """Rough cardinal direction from one point to another."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    angle = math.degrees(math.atan2(dy, dx)) % 360
    directions = [
        (0, "east"), (45, "northeast"), (90, "north"),
        (135, "northwest"), (180, "west"), (225, "southwest"),
        (270, "south"), (315, "southeast"), (360, "east"),
    ]
    closest = min(directions, key=lambda d: abs(d[0] - angle))
    return closest[1]


def _dist_word(d: float) -> str:
    """Rough distance description."""
    if d < 20:
        return "danger close"
    if d < 50:
        return "close range"
    if d < 150:
        return "medium range"
    if d < 400:
        return "long range"
    return "extreme range"


def _unit_label(unit: Any) -> str:
    """Best effort label for a unit-like object."""
    if hasattr(unit, "name") and unit.name:
        return str(unit.name)
    if hasattr(unit, "unit_id"):
        return str(unit.unit_id)
    return str(unit)


def _pick(templates: list[str]) -> str:
    """Pick a random template string."""
    return random.choice(templates)


# ---------------------------------------------------------------------------
# BattleNarrator
# ---------------------------------------------------------------------------

class BattleNarrator:
    """Generates varied military-style radio chatter from sim events.

    Each ``narrate_*`` method returns a :class:`NarrationEvent` with
    randomized text so repeated events sound different.  The active
    :class:`CommanderPersonality` influences word choice and detail level.
    """

    def __init__(
        self,
        personality: CommanderPersonality | None = None,
        tick: int = 0,
        sim_time: float = 0.0,
    ) -> None:
        self.personality = personality or PERSONALITIES["iron_hand"]
        self._tick = tick
        self._sim_time = sim_time

    # -- state updates ---
    def set_tick(self, tick: int, sim_time: float) -> None:
        """Advance the narrator's clock."""
        self._tick = tick
        self._sim_time = sim_time

    # -- narration methods --------------------------------------------------

    def narrate_kill(self, killer: Any, victim: Any) -> NarrationEvent:
        """Narrate a confirmed kill."""
        kl = _unit_label(killer)
        vl = _unit_label(victim)
        cs = self.personality.callsign

        templates = [
            f"{cs}, {kl} confirms hostile down. Target {vl} eliminated.",
            f"{kl} confirms kill on {vl}. Tango down.",
            f"Splash one. {kl} neutralized {vl}.",
            f"{cs} actual, {kl} reports hostile {vl} is KIA.",
            f"Good hit, good hit. {vl} is down. {kl} confirms.",
            f"{kl} — target destroyed. {vl} is no longer a threat.",
        ]
        if self.personality.style == "aggressive":
            templates.extend([
                f"That's another one! {kl} just dusted {vl}!",
                f"{kl} sends {vl} to the shadow realm!",
            ])
        if self.personality.style == "dramatic":
            templates.extend([
                f"And {kl} delivers the killing blow! {vl} goes down hard!",
                f"What a shot! {kl} takes out {vl} with surgical precision!",
            ])
        if self.personality.humor > 0.3:
            templates.append(f"{kl} just retired {vl}. Permanently.")

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="combat",
            priority=2,
            text=_pick(templates),
            voice="radio",
        )

    def narrate_engagement(self, attacker: Any, target: Any, result: str) -> NarrationEvent:
        """Narrate a combat engagement (hit, miss, suppression)."""
        al = _unit_label(attacker)
        tl = _unit_label(target)
        cs = self.personality.callsign

        if result == "hit":
            templates = [
                f"Contact! {al} engaging {tl} — hit confirmed!",
                f"{al} lands a hit on {tl}. Target still active.",
                f"{cs}, {al} reports good effect on {tl}.",
                f"Rounds on target. {al} scoring hits on {tl}.",
            ]
        elif result == "miss":
            templates = [
                f"{al} engaging {tl} — rounds going wide.",
                f"Negative effect. {al} missed {tl}.",
                f"{cs}, {al} reports no joy on {tl}. Adjusting fire.",
                f"{al} firing on {tl}, no hits yet.",
            ]
        elif result == "suppression":
            templates = [
                f"{al} putting suppressive fire on {tl}.",
                f"Suppressing! {al} keeping {tl} pinned down.",
                f"{cs}, {al} has {tl} suppressed.",
                f"{al} laying down cover fire on {tl}'s position.",
            ]
        else:
            templates = [
                f"{al} engaging {tl}. Result: {result}.",
                f"Contact between {al} and {tl}. Status: {result}.",
            ]

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="combat",
            priority=2 if result == "hit" else 1,
            text=_pick(templates),
            voice="radio",
        )

    def narrate_explosion(
        self, position: Vec2, radius: float, casualties: int
    ) -> NarrationEvent:
        """Narrate an explosion event."""
        grid = _grid_ref(position)
        cs = self.personality.callsign

        if casualties == 0:
            templates = [
                f"Explosion at grid {grid}. No casualties reported.",
                f"{cs}, detonation at {grid}. Area clear.",
                f"Blast at {grid}, {radius:.0f}m radius. No effect on personnel.",
            ]
            priority = 2
        elif casualties == 1:
            templates = [
                f"Explosion at grid {grid}! One casualty!",
                f"{cs}, blast at {grid} — we have one down!",
                f"Detonation at {grid}, {radius:.0f}m radius. One casualty confirmed.",
            ]
            priority = 3
        else:
            templates = [
                f"Major explosion at grid {grid}! {casualties} casualties!",
                f"{cs}, mass-cas event at {grid}! {casualties} down!",
                f"Detonation at {grid}, {radius:.0f}m blast radius. {casualties} casualties!",
                f"BOOM at {grid}! Multiple casualties — count is {casualties}!",
            ]
            priority = 4

        if self.personality.style == "dramatic" and casualties > 0:
            templates.append(
                f"The ground shakes at {grid}! {casualties} souls caught in the blast!"
            )

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="combat",
            priority=priority,
            text=_pick(templates),
            voice="radio",
        )

    def narrate_movement(
        self, unit: Any, from_pos: Vec2, to_pos: Vec2, action: str = "moving"
    ) -> NarrationEvent:
        """Narrate a unit movement."""
        ul = _unit_label(unit)
        bearing = _bearing_word(from_pos, to_pos)
        dist = distance(from_pos, to_pos)
        grid = _grid_ref(to_pos)

        templates = [
            f"{ul} {action} {bearing}, {dist:.0f} meters to grid {grid}.",
            f"{self.personality.callsign}, {ul} is {action} {bearing}. Destination: {grid}.",
            f"{ul} relocating {bearing}. ETA to {grid}: {dist/5.0:.0f} seconds.",
        ]
        if action == "retreating":
            templates.extend([
                f"{ul} falling back {bearing}! Heading to {grid}!",
                f"Pull back! {ul} retreating {bearing}!",
            ])
        elif action == "advancing":
            templates.extend([
                f"{ul} pushing {bearing} toward grid {grid}.",
                f"Moving up! {ul} advancing {bearing}.",
            ])

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="tactical",
            priority=2 if action == "retreating" else 1,
            text=_pick(templates),
            voice="radio",
        )

    def narrate_wave_start(self, wave_num: int, hostiles: int) -> NarrationEvent:
        """Narrate the start of a combat wave."""
        cs = self.personality.callsign

        templates = [
            f"{cs}, Wave {wave_num} incoming! {hostiles} hostiles detected!",
            f"All stations, be advised: Wave {wave_num} — {hostiles} hostiles inbound!",
            f"Wave {wave_num} has started. Intel counts {hostiles} enemy contacts.",
            f"Heads up! Wave {wave_num} is live. {hostiles} tangos approaching!",
        ]
        if self.personality.style == "aggressive":
            templates.append(f"Here they come! Wave {wave_num} — {hostiles} targets! Let's go!")
        if self.personality.style == "dramatic":
            templates.append(
                f"The storm begins! Wave {wave_num} unleashes {hostiles} hostiles upon us!"
            )

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="combat",
            priority=3,
            text=_pick(templates),
            voice="commander",
        )

    def narrate_wave_clear(self, wave_num: int, time_taken: float) -> NarrationEvent:
        """Narrate the completion of a combat wave."""
        cs = self.personality.callsign

        templates = [
            f"{cs}, Wave {wave_num} clear. Time: {time_taken:.1f}s.",
            f"All hostiles neutralized. Wave {wave_num} complete in {time_taken:.1f} seconds.",
            f"Wave {wave_num} cleared. Area secure. Elapsed: {time_taken:.1f}s.",
        ]
        if self.personality.style == "aggressive":
            templates.append(f"Wave {wave_num} crushed in {time_taken:.1f}s! Bring on the next one!")
        if self.personality.style == "dramatic":
            templates.append(
                f"Victory! Wave {wave_num} falls after {time_taken:.1f} seconds of glorious combat!"
            )
        if self.personality.humor > 0.3:
            templates.append(f"Wave {wave_num} down in {time_taken:.1f}s. Too easy.")

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="combat",
            priority=2,
            text=_pick(templates),
            voice="commander",
        )

    def narrate_casualty(self, unit: Any, injury: str) -> NarrationEvent:
        """Narrate a friendly casualty."""
        ul = _unit_label(unit)
        cs = self.personality.callsign

        templates = [
            f"Man down! {ul} hit — {injury}!",
            f"{cs}, {ul} is wounded! {injury}! Requesting medic!",
            f"Casualty report: {ul} sustained {injury}.",
            f"Medic! {ul} is hit — {injury}! Need immediate assistance!",
            f"{ul} took a hit. Injury: {injury}. Still in the fight.",
        ]
        if self.personality.style == "cautious":
            templates.append(f"Be advised, {ul} is down with {injury}. Pulling back to treat.")

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="medical",
            priority=3,
            text=_pick(templates),
            voice="medic",
        )

    def narrate_supply_low(self, supply_type: str, level: float) -> NarrationEvent:
        """Narrate a low supply warning."""
        cs = self.personality.callsign
        pct = int(level * 100)

        templates = [
            f"{cs}, {supply_type} at {pct}%. Request resupply.",
            f"Low on {supply_type}! Only {pct}% remaining.",
            f"Supply warning: {supply_type} down to {pct}%.",
            f"Running low on {supply_type}. {pct}% left. Need resupply ASAP.",
        ]
        priority = 3 if level < 0.1 else 2

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="logistics",
            priority=priority,
            text=_pick(templates),
            voice="radio",
        )

    def narrate_detection(self, sensor_type: str, target: Any) -> NarrationEvent:
        """Narrate a new sensor detection."""
        tl = _unit_label(target)
        cs = self.personality.callsign

        templates = [
            f"{cs}, new contact on {sensor_type}. Designating {tl}.",
            f"{sensor_type} pick-up: new contact {tl}.",
            f"Sensor alert — {sensor_type} detected {tl}.",
            f"Contact! {sensor_type} has eyes on {tl}.",
        ]
        if self.personality.style == "cautious":
            templates.append(f"Unknown contact on {sensor_type}: {tl}. Proceed with caution.")

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="intel",
            priority=2,
            text=_pick(templates),
            voice="observer",
        )

    def narrate_weather_change(self, old: str, new: str) -> NarrationEvent:
        """Narrate a weather condition change."""
        cs = self.personality.callsign

        templates = [
            f"{cs}, weather shifting from {old} to {new}.",
            f"Weather update: {old} clearing, {new} moving in.",
            f"All stations, weather change — transitioning from {old} to {new}.",
            f"Meteorological advisory: {new} conditions replacing {old}.",
        ]
        if self.personality.style == "dramatic":
            templates.append(f"The skies darken as {old} gives way to {new}!")

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="tactical",
            priority=1,
            text=_pick(templates),
            voice="observer",
        )

    def narrate_achievement(self, unit: Any, achievement: str) -> NarrationEvent:
        """Narrate a unit achievement."""
        ul = _unit_label(unit)
        cs = self.personality.callsign

        templates = [
            f"Outstanding! {ul} earned: {achievement}!",
            f"{cs}, commendation for {ul} — {achievement}.",
            f"{ul} just unlocked {achievement}. Well done.",
        ]
        if self.personality.style == "dramatic":
            templates.append(f"Legend! {ul} achieves the legendary {achievement}!")
        if self.personality.style == "aggressive":
            templates.append(f"Hell yeah! {ul} earned {achievement}! Keep it up!")
        if self.personality.humor > 0.3:
            templates.append(f"{ul} just got {achievement}. Not bad for a Tuesday.")

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="combat",
            priority=1,
            text=_pick(templates),
            voice="commander",
        )

    def narrate_game_over(self, winner: str, stats: dict[str, Any]) -> NarrationEvent:
        """Narrate end-of-game summary."""
        cs = self.personality.callsign
        kills = stats.get("total_kills", 0)
        casualties = stats.get("total_casualties", 0)
        duration = stats.get("duration", 0.0)

        templates = [
            (
                f"Battle complete. {winner} victorious. "
                f"Kills: {kills}, Casualties: {casualties}, Duration: {duration:.0f}s."
            ),
            (
                f"{cs}, end of engagement. Winner: {winner}. "
                f"Final tally — {kills} kills, {casualties} losses over {duration:.0f} seconds."
            ),
            (
                f"All stations, ceasefire. {winner} wins. "
                f"Stats: {kills} hostiles eliminated, {casualties} friendly casualties, {duration:.0f}s."
            ),
        ]
        if self.personality.style == "dramatic":
            templates.append(
                f"And so it ends! {winner} stands triumphant with {kills} kills "
                f"and only {casualties} lost in {duration:.0f} seconds of combat!"
            )
        if self.personality.style == "aggressive":
            templates.append(
                f"Done! {winner} dominates — {kills} kills, {casualties} down, {duration:.0f}s. GG."
            )

        return NarrationEvent(
            tick=self._tick,
            time=self._sim_time,
            category="combat",
            priority=4,
            text=_pick(templates),
            voice="commander",
        )


# ---------------------------------------------------------------------------
# TacticalAdvisor
# ---------------------------------------------------------------------------

class TacticalAdvisor:
    """Generates tactical assessments and recommendations from world state."""

    def __init__(self, personality: CommanderPersonality | None = None) -> None:
        self.personality = personality or PERSONALITIES["iron_hand"]

    def assess_situation(self, world_state: dict[str, Any]) -> list[str]:
        """Generate tactical recommendations from world state.

        ``world_state`` should contain keys like ``friendly_count``,
        ``hostile_count``, ``friendly_casualties``, ``ammo_level``,
        ``visibility``, etc.
        """
        recs: list[str] = []
        friendly = world_state.get("friendly_count", 0)
        hostile = world_state.get("hostile_count", 0)
        casualties = world_state.get("friendly_casualties", 0)
        ammo = world_state.get("ammo_level", 1.0)
        visibility = world_state.get("visibility", 1.0)

        # Force ratio
        if hostile > 0 and friendly > 0:
            ratio = friendly / hostile
            if ratio < 0.5:
                recs.append("CRITICAL: Heavily outnumbered. Recommend immediate withdrawal or reinforcement.")
            elif ratio < 1.0:
                recs.append("Outnumbered. Recommend defensive posture and use of cover.")
            elif ratio > 3.0:
                recs.append("Overwhelming advantage. Press the attack.")
            else:
                recs.append("Force parity. Maintain current engagement strategy.")

        # Casualty rate
        if friendly > 0 and casualties > 0:
            cas_rate = casualties / (friendly + casualties)
            if cas_rate > 0.5:
                recs.append("CRITICAL: Over 50% casualties. Recommend fallback to rally point.")
            elif cas_rate > 0.25:
                recs.append("Heavy casualties. Consider consolidating remaining forces.")

        # Ammo
        if ammo < 0.2:
            recs.append("Ammunition critical. Switch to conservation fire or request resupply.")
        elif ammo < 0.5:
            recs.append("Ammunition below 50%. Prioritize targets carefully.")

        # Visibility
        if visibility < 0.3:
            recs.append("Low visibility. Expect reduced accuracy. Use sensors and thermal.")
        elif visibility < 0.6:
            recs.append("Reduced visibility. Close engagement ranges recommended.")

        # No hostiles
        if hostile == 0 and friendly > 0:
            recs.append("Area clear. Establish perimeter and await next wave.")

        return recs

    def recommend_action(self, unit_id: str, situation: dict[str, Any]) -> str:
        """Recommend an action for a specific unit.

        ``situation`` may contain ``health``, ``ammo``, ``enemies_visible``,
        ``nearest_cover_dist``, ``in_cover``, ``suppressed``.
        """
        health = situation.get("health", 100.0)
        ammo = situation.get("ammo", -1)
        enemies = situation.get("enemies_visible", 0)
        cover_dist = situation.get("nearest_cover_dist", 999.0)
        in_cover = situation.get("in_cover", False)
        suppressed = situation.get("suppressed", False)

        if health < 20:
            return f"{unit_id}: Fall back to medic. Health critical."
        if suppressed:
            if not in_cover and cover_dist < 30:
                return f"{unit_id}: Sprint to nearest cover ({cover_dist:.0f}m)."
            return f"{unit_id}: Stay low. Wait for suppression to lift."
        if ammo == 0:
            return f"{unit_id}: Winchester. Fall back for resupply."
        if enemies == 0:
            return f"{unit_id}: No contacts. Hold position and scan."
        if not in_cover and cover_dist < 20:
            return f"{unit_id}: Get to cover ({cover_dist:.0f}m) before engaging."
        if enemies >= 3 and self.personality.style == "cautious":
            return f"{unit_id}: Multiple contacts. Hold fire, wait for support."
        return f"{unit_id}: Engage nearest hostile."

    def threat_warning(self, threats: list[dict[str, Any]]) -> str:
        """Generate a threat warning from a list of threat dicts.

        Each threat should have ``type``, ``bearing``, ``distance``, and
        optionally ``count``.
        """
        if not threats:
            return "No active threats detected."

        parts: list[str] = []
        for t in threats:
            ttype = t.get("type", "unknown")
            bearing = t.get("bearing", "unknown")
            dist = t.get("distance", 0)
            count = t.get("count", 1)
            if count > 1:
                parts.append(f"{count}x {ttype} bearing {bearing}, {dist:.0f}m")
            else:
                parts.append(f"{ttype} bearing {bearing}, {dist:.0f}m")

        cs = self.personality.callsign
        return f"THREAT WARNING from {cs}: {'; '.join(parts)}."

    def sitrep(self, world_state: dict[str, Any]) -> str:
        """Generate a situation report paragraph."""
        friendly = world_state.get("friendly_count", 0)
        hostile = world_state.get("hostile_count", 0)
        casualties = world_state.get("friendly_casualties", 0)
        wave = world_state.get("current_wave", 0)
        elapsed = world_state.get("elapsed_time", 0.0)
        ammo = world_state.get("ammo_level", 1.0)
        cs = self.personality.callsign

        lines = [f"SITREP from {cs} at T+{elapsed:.0f}s:"]
        if wave > 0:
            lines.append(f"Current wave: {wave}.")
        lines.append(f"Friendly strength: {friendly} active, {casualties} casualties.")
        if hostile > 0:
            lines.append(f"Enemy contacts: {hostile} remaining.")
        else:
            lines.append("No enemy contacts.")
        lines.append(f"Ammunition: {int(ammo * 100)}%.")

        # Overall assessment
        if hostile == 0:
            lines.append("Assessment: Area secure. Awaiting next engagement.")
        elif friendly > hostile * 2:
            lines.append("Assessment: Favorable conditions. Maintain offensive pressure.")
        elif hostile > friendly * 2:
            lines.append("Assessment: Unfavorable odds. Defensive posture recommended.")
        else:
            lines.append("Assessment: Contested. Exercise caution.")

        return " ".join(lines)

    def battle_summary(self, aar: dict[str, Any]) -> str:
        """Generate an after-action report narrative.

        ``aar`` should contain ``winner``, ``duration``, ``waves_completed``,
        ``total_kills``, ``total_casualties``, ``accuracy``, etc.
        """
        winner = aar.get("winner", "unknown")
        duration = aar.get("duration", 0.0)
        waves = aar.get("waves_completed", 0)
        kills = aar.get("total_kills", 0)
        casualties = aar.get("total_casualties", 0)
        accuracy = aar.get("accuracy", 0.0)
        cs = self.personality.callsign

        lines = [f"AFTER-ACTION REPORT — {cs}"]
        lines.append(f"Engagement duration: {duration:.0f} seconds across {waves} waves.")
        lines.append(f"Result: {winner} victory.")
        lines.append(f"Enemy eliminated: {kills}. Friendly losses: {casualties}.")
        if accuracy > 0:
            lines.append(f"Overall accuracy: {accuracy * 100:.1f}%.")

        # Verdict
        if casualties == 0:
            lines.append("Assessment: Flawless engagement. No friendly losses.")
        elif casualties <= kills * 0.1:
            lines.append("Assessment: Decisive victory with minimal casualties.")
        elif casualties <= kills * 0.5:
            lines.append("Assessment: Victory achieved at moderate cost.")
        else:
            lines.append("Assessment: Pyrrhic victory. Casualty rate unacceptable.")

        return " ".join(lines)


# ---------------------------------------------------------------------------
# NarrationLog
# ---------------------------------------------------------------------------

class NarrationLog:
    """Accumulates narration events and renders them for UI display.

    Provides filtering by priority and category, and a ``to_three_js()``
    export suitable for a Three.js HUD overlay.
    """

    def __init__(self, max_events: int = 500) -> None:
        self._events: list[NarrationEvent] = []
        self._max_events = max_events

    def add(self, event: NarrationEvent) -> None:
        """Add a narration event to the log."""
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    @property
    def events(self) -> list[NarrationEvent]:
        """All accumulated events."""
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def filter_by_priority(self, min_priority: int) -> list[NarrationEvent]:
        """Return events at or above *min_priority*."""
        return [e for e in self._events if e.priority >= min_priority]

    def filter_by_category(self, category: str) -> list[NarrationEvent]:
        """Return events matching *category*."""
        return [e for e in self._events if e.category == category]

    def recent(self, count: int = 10) -> list[NarrationEvent]:
        """Return the most recent *count* events."""
        return self._events[-count:]

    def clear(self) -> None:
        """Remove all events."""
        self._events.clear()

    def to_three_js(self) -> dict[str, Any]:
        """Export narration log in a format suitable for Three.js HUD.

        Returns a dict with:
            kill_feed  — last 5 combat events
            alerts     — priority 3+ events from last 10
            sitrep     — last tactical/intel event text or empty
        """
        combat = [e for e in self._events if e.category == "combat"]
        kill_feed = [
            {"tick": e.tick, "text": e.text, "priority": e.priority}
            for e in combat[-5:]
        ]

        recent = self._events[-10:]
        alerts = [
            {"tick": e.tick, "text": e.text, "priority": e.priority, "voice": e.voice}
            for e in recent
            if e.priority >= 3
        ]

        # Last tactical or intel event
        sitrep_text = ""
        for e in reversed(self._events):
            if e.category in ("tactical", "intel"):
                sitrep_text = e.text
                break

        return {
            "kill_feed": kill_feed,
            "alerts": alerts,
            "sitrep": sitrep_text,
        }

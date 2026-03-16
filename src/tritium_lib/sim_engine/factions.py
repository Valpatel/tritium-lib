# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Multi-faction diplomacy and relationship engine.

Simulates dynamic faction relationships: alliances, betrayals, negotiations,
ceasefires, and trust evolution.  Each faction has ideology, strength, wealth,
territory, and morale.  Relationships evolve based on in-sim events (attacks,
collateral damage, shared enemies, broken ceasefires).

Usage::

    engine = DiplomacyEngine()
    engine.add_faction(Faction(faction_id="gov", name="Government", color="#00ff00",
                               ideology="government", strength=0.8, wealth=0.7))
    engine.add_faction(Faction(faction_id="reb", name="Rebels", color="#ff0000",
                               ideology="rebel", strength=0.4, wealth=0.2))
    engine.declare_war("gov", "reb")
    assert engine.are_hostile("gov", "reb")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tritium_lib.sim_engine.ai.steering import Vec2

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Relation(Enum):
    """Diplomatic relationship level between two factions, ordered from
    friendliest to most hostile."""

    ALLIED = "allied"
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    SUSPICIOUS = "suspicious"
    HOSTILE = "hostile"
    WAR = "war"


# Numeric severity for ordering and arithmetic.
_RELATION_ORDER: dict[Relation, int] = {
    Relation.ALLIED: 0,
    Relation.FRIENDLY: 1,
    Relation.NEUTRAL: 2,
    Relation.SUSPICIOUS: 3,
    Relation.HOSTILE: 4,
    Relation.WAR: 5,
}

_ORDER_TO_RELATION: dict[int, Relation] = {v: k for k, v in _RELATION_ORDER.items()}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Faction:
    """A faction participating in the simulation."""

    faction_id: str
    name: str
    color: str  # hex for Three.js, e.g. "#ff2a6d"
    ideology: str  # government, rebel, criminal, mercenary, civilian, peacekeeping
    strength: float  # 0-1 military power
    wealth: float  # 0-1 resources
    territory: list[Vec2] = field(default_factory=list)
    leader: str | None = None
    morale: float = 0.7


@dataclass
class DiplomaticRelation:
    """Relationship state between two factions."""

    faction_a: str
    faction_b: str
    relation: Relation = Relation.NEUTRAL
    trust: float = 0.5  # 0-1
    trade_active: bool = False
    ceasefire_until: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def _log(self, event_type: str, detail: str, sim_time: float = 0.0) -> None:
        self.history.append({
            "type": event_type,
            "detail": detail,
            "sim_time": sim_time,
            "relation": self.relation.value,
            "trust": round(self.trust, 3),
        })


# ---------------------------------------------------------------------------
# DiplomacyEngine
# ---------------------------------------------------------------------------


class DiplomacyEngine:
    """Central diplomacy simulation.

    Manages factions, their pairwise relations, and the event-driven tick
    that evolves trust and relations over time.
    """

    def __init__(self) -> None:
        self.factions: dict[str, Faction] = {}
        self.relations: dict[tuple[str, str], DiplomaticRelation] = {}
        self._unit_faction: dict[str, str] = {}  # unit_id -> faction_id
        self._sim_time: float = 0.0

    # -- faction management -------------------------------------------------

    def add_faction(self, faction: Faction) -> None:
        """Register a faction.  Automatically creates NEUTRAL relations with
        all existing factions."""
        self.factions[faction.faction_id] = faction
        if faction.leader:
            self._unit_faction[faction.leader] = faction.faction_id
        for fid in self.factions:
            if fid != faction.faction_id:
                key = self._key(faction.faction_id, fid)
                if key not in self.relations:
                    self.relations[key] = DiplomaticRelation(
                        faction_a=key[0], faction_b=key[1]
                    )

    def remove_faction(self, faction_id: str) -> None:
        """Remove a faction and all its relations."""
        self.factions.pop(faction_id, None)
        to_del = [k for k in self.relations if faction_id in k]
        for k in to_del:
            del self.relations[k]
        to_del_units = [u for u, f in self._unit_faction.items() if f == faction_id]
        for u in to_del_units:
            del self._unit_faction[u]

    def assign_unit(self, unit_id: str, faction_id: str) -> None:
        """Assign a unit to a faction."""
        self._unit_faction[unit_id] = faction_id

    def faction_for_unit(self, unit_id: str) -> str | None:
        """Return the faction_id a unit belongs to, or None."""
        return self._unit_faction.get(unit_id)

    # -- relation lookup / mutation -----------------------------------------

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        """Canonical ordered key so (a,b) == (b,a)."""
        return (min(a, b), max(a, b))

    def get_relation(self, a: str, b: str) -> DiplomaticRelation:
        """Get the diplomatic relation between two factions.  Creates a
        NEUTRAL one if it doesn't exist yet."""
        key = self._key(a, b)
        if key not in self.relations:
            self.relations[key] = DiplomaticRelation(
                faction_a=key[0], faction_b=key[1]
            )
        return self.relations[key]

    def set_relation(self, a: str, b: str, relation: Relation) -> None:
        """Change the relationship between two factions, logging history."""
        dr = self.get_relation(a, b)
        old = dr.relation
        dr.relation = relation
        dr._log("relation_change", f"{old.value} -> {relation.value}",
                self._sim_time)
        # War cancels trade
        if relation == Relation.WAR:
            dr.trade_active = False

    def declare_war(self, a: str, b: str) -> None:
        """Declare war between two factions."""
        dr = self.get_relation(a, b)
        dr.relation = Relation.WAR
        dr.trust = max(0.0, dr.trust - 0.4)
        dr.trade_active = False
        dr.ceasefire_until = None
        dr._log("war_declared", f"{a} declared war on {b}", self._sim_time)

    def propose_ceasefire(self, a: str, b: str, duration: float) -> bool:
        """Propose a ceasefire.  Accepted if trust > 0.1.  Returns True if
        accepted."""
        dr = self.get_relation(a, b)
        if dr.trust <= 0.1:
            dr._log("ceasefire_rejected", "trust too low", self._sim_time)
            return False
        dr.ceasefire_until = self._sim_time + duration
        dr.relation = Relation.HOSTILE  # still hostile, just not shooting
        dr._log("ceasefire_accepted", f"duration={duration}", self._sim_time)
        return True

    def form_alliance(self, a: str, b: str) -> None:
        """Form an alliance between two factions."""
        dr = self.get_relation(a, b)
        dr.relation = Relation.ALLIED
        dr.trust = min(1.0, dr.trust + 0.3)
        dr.trade_active = True
        dr._log("alliance_formed", f"{a} allied with {b}", self._sim_time)

    def break_alliance(self, a: str, b: str) -> None:
        """Break an existing alliance.  Trust drops sharply."""
        dr = self.get_relation(a, b)
        if dr.relation == Relation.ALLIED:
            dr.relation = Relation.SUSPICIOUS
            dr.trust = max(0.0, dr.trust - 0.4)
            dr.trade_active = False
            dr._log("alliance_broken", f"{a} broke alliance with {b}",
                     self._sim_time)

    def are_hostile(self, a: str, b: str) -> bool:
        """True if the two factions are HOSTILE or at WAR."""
        dr = self.get_relation(a, b)
        return dr.relation in (Relation.HOSTILE, Relation.WAR)

    def are_allied(self, a: str, b: str) -> bool:
        """True if the two factions are ALLIED."""
        dr = self.get_relation(a, b)
        return dr.relation == Relation.ALLIED

    # -- tick ---------------------------------------------------------------

    def tick(self, dt: float, events: list[dict[str, Any]] | None = None) -> None:
        """Advance diplomacy by *dt* seconds, processing events.

        Supported event types (dict keys):
            ``{"type": "attack", "attacker_faction": str, "target_faction": str}``
            ``{"type": "collateral", "attacker_faction": str, "victim_faction": str}``
            ``{"type": "shared_enemy", "faction_a": str, "faction_b": str, "enemy": str}``
            ``{"type": "ceasefire_broken", "breaker": str, "victim": str}``
            ``{"type": "trade", "faction_a": str, "faction_b": str}``
        """
        self._sim_time += dt
        if not events:
            return
        for ev in events:
            etype = ev.get("type", "")
            if etype == "attack":
                self._handle_attack(ev)
            elif etype == "collateral":
                self._handle_collateral(ev)
            elif etype == "shared_enemy":
                self._handle_shared_enemy(ev)
            elif etype == "ceasefire_broken":
                self._handle_ceasefire_broken(ev)
            elif etype == "trade":
                self._handle_trade(ev)

    def _handle_attack(self, ev: dict[str, Any]) -> None:
        """Attack on a faction degrades the relationship."""
        a = ev["attacker_faction"]
        b = ev["target_faction"]
        dr = self.get_relation(a, b)
        dr.trust = max(0.0, dr.trust - 0.15)
        # Escalate relation
        cur = _RELATION_ORDER[dr.relation]
        new_ord = min(cur + 1, 5)
        dr.relation = _ORDER_TO_RELATION[new_ord]
        dr._log("attack", f"{a} attacked {b}", self._sim_time)
        if dr.relation == Relation.WAR:
            dr.trade_active = False

    def _handle_collateral(self, ev: dict[str, Any]) -> None:
        """Collateral damage (especially to civilians) tanks trust hard."""
        a = ev["attacker_faction"]
        b = ev["victim_faction"]
        dr = self.get_relation(a, b)
        victim_faction = self.factions.get(b)
        penalty = 0.25 if victim_faction and victim_faction.ideology == "civilian" else 0.15
        dr.trust = max(0.0, dr.trust - penalty)
        # Degrade relation by 1 step
        cur = _RELATION_ORDER[dr.relation]
        new_ord = min(cur + 1, 5)
        dr.relation = _ORDER_TO_RELATION[new_ord]
        dr._log("collateral", f"{a} caused collateral damage to {b}",
                 self._sim_time)

    def _handle_shared_enemy(self, ev: dict[str, Any]) -> None:
        """Factions that share an enemy grow closer."""
        a = ev["faction_a"]
        b = ev["faction_b"]
        dr = self.get_relation(a, b)
        dr.trust = min(1.0, dr.trust + 0.1)
        # Improve relation by 1 step (but not past FRIENDLY — alliance requires explicit action)
        cur = _RELATION_ORDER[dr.relation]
        new_ord = max(cur - 1, 1)  # floor at FRIENDLY
        dr.relation = _ORDER_TO_RELATION[new_ord]
        dr._log("shared_enemy", f"{a} and {b} share enemy {ev.get('enemy', '?')}",
                 self._sim_time)

    def _handle_ceasefire_broken(self, ev: dict[str, Any]) -> None:
        """Breaking a ceasefire is the ultimate betrayal: instant WAR, trust 0."""
        breaker = ev["breaker"]
        victim = ev["victim"]
        dr = self.get_relation(breaker, victim)
        dr.relation = Relation.WAR
        dr.trust = 0.0
        dr.ceasefire_until = None
        dr.trade_active = False
        dr._log("ceasefire_broken", f"{breaker} broke ceasefire with {victim}",
                 self._sim_time)

    def _handle_trade(self, ev: dict[str, Any]) -> None:
        """Active trade between factions improves trust slightly."""
        a = ev["faction_a"]
        b = ev["faction_b"]
        dr = self.get_relation(a, b)
        if dr.relation not in (Relation.HOSTILE, Relation.WAR):
            dr.trust = min(1.0, dr.trust + 0.05)
            dr.trade_active = True
            dr._log("trade", f"trade between {a} and {b}", self._sim_time)

    # -- query / export -----------------------------------------------------

    def get_enemies(self, faction_id: str) -> list[str]:
        """Return all faction IDs hostile or at war with *faction_id*."""
        enemies: list[str] = []
        for (a, b), dr in self.relations.items():
            if dr.relation in (Relation.HOSTILE, Relation.WAR):
                if a == faction_id:
                    enemies.append(b)
                elif b == faction_id:
                    enemies.append(a)
        return enemies

    def get_allies(self, faction_id: str) -> list[str]:
        """Return all faction IDs allied with *faction_id*."""
        allies: list[str] = []
        for (a, b), dr in self.relations.items():
            if dr.relation == Relation.ALLIED:
                if a == faction_id:
                    allies.append(b)
                elif b == faction_id:
                    allies.append(a)
        return allies

    def get_diplomatic_map(self) -> dict[str, Any]:
        """Full snapshot for visualization: all factions and relations."""
        return {
            "factions": {
                fid: {
                    "name": f.name,
                    "color": f.color,
                    "ideology": f.ideology,
                    "strength": f.strength,
                    "wealth": f.wealth,
                    "morale": f.morale,
                    "territory_count": len(f.territory),
                    "leader": f.leader,
                }
                for fid, f in self.factions.items()
            },
            "relations": [
                {
                    "faction_a": dr.faction_a,
                    "faction_b": dr.faction_b,
                    "relation": dr.relation.value,
                    "trust": round(dr.trust, 3),
                    "trade_active": dr.trade_active,
                    "ceasefire_until": dr.ceasefire_until,
                    "history_len": len(dr.history),
                }
                for dr in self.relations.values()
            ],
            "sim_time": self._sim_time,
        }

    def to_three_js(self) -> dict[str, Any]:
        """Export for Three.js visualization — faction territories as colored
        polygons, relations as colored lines between territory centroids."""
        factions_js: list[dict[str, Any]] = []
        for fid, f in self.factions.items():
            centroid = _centroid(f.territory) if f.territory else (0.0, 0.0)
            factions_js.append({
                "id": fid,
                "name": f.name,
                "color": f.color,
                "centroid": list(centroid),
                "territory": [list(p) for p in f.territory],
                "strength": f.strength,
                "morale": f.morale,
            })

        relation_lines: list[dict[str, Any]] = []
        for (a, b), dr in self.relations.items():
            fa = self.factions.get(a)
            fb = self.factions.get(b)
            if not fa or not fb:
                continue
            ca = _centroid(fa.territory) if fa.territory else (0.0, 0.0)
            cb = _centroid(fb.territory) if fb.territory else (0.0, 0.0)
            relation_lines.append({
                "from": a,
                "to": b,
                "from_pos": list(ca),
                "to_pos": list(cb),
                "relation": dr.relation.value,
                "color": _relation_color(dr.relation),
                "trust": round(dr.trust, 3),
            })

        return {
            "factions": factions_js,
            "relations": relation_lines,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centroid(points: list[Vec2]) -> Vec2:
    """Simple centroid of a list of 2D points."""
    if not points:
        return (0.0, 0.0)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    n = len(points)
    return (sx / n, sy / n)


_RELATION_COLORS: dict[Relation, str] = {
    Relation.ALLIED: "#05ffa1",    # green
    Relation.FRIENDLY: "#00f0ff",  # cyan
    Relation.NEUTRAL: "#888888",   # grey
    Relation.SUSPICIOUS: "#fcee0a",  # yellow
    Relation.HOSTILE: "#ff8800",   # orange
    Relation.WAR: "#ff2a6d",       # magenta/red
}


def _relation_color(r: Relation) -> str:
    return _RELATION_COLORS.get(r, "#ffffff")


# ---------------------------------------------------------------------------
# Faction presets
# ---------------------------------------------------------------------------

def _make_faction(fid: str, name: str, color: str, ideology: str,
                  strength: float, wealth: float, morale: float = 0.7) -> Faction:
    return Faction(
        faction_id=fid, name=name, color=color, ideology=ideology,
        strength=strength, wealth=wealth, morale=morale,
    )


FACTION_PRESETS: dict[str, dict[str, Any]] = {
    "three_way_war": {
        "description": "Three hostile factions in total war",
        "factions": [
            _make_faction("alpha", "Alpha Corps", "#ff2a6d", "mercenary", 0.7, 0.5),
            _make_faction("bravo", "Bravo Brigade", "#00f0ff", "mercenary", 0.6, 0.6),
            _make_faction("charlie", "Charlie Company", "#05ffa1", "mercenary", 0.5, 0.4),
        ],
        "initial_relations": [
            ("alpha", "bravo", Relation.WAR),
            ("alpha", "charlie", Relation.WAR),
            ("bravo", "charlie", Relation.WAR),
        ],
    },
    "proxy_conflict": {
        "description": "Two superpowers backing proxy factions",
        "factions": [
            _make_faction("east", "Eastern Alliance", "#ff2a6d", "government", 0.9, 0.9),
            _make_faction("west", "Western Coalition", "#00f0ff", "government", 0.9, 0.9),
            _make_faction("proxy_e", "Eastern Proxy", "#ff8800", "rebel", 0.4, 0.2),
            _make_faction("proxy_w", "Western Proxy", "#0088ff", "rebel", 0.4, 0.2),
        ],
        "initial_relations": [
            ("east", "west", Relation.HOSTILE),
            ("east", "proxy_e", Relation.ALLIED),
            ("west", "proxy_w", Relation.ALLIED),
            ("east", "proxy_w", Relation.HOSTILE),
            ("west", "proxy_e", Relation.HOSTILE),
            ("proxy_e", "proxy_w", Relation.WAR),
        ],
    },
    "insurgency": {
        "description": "Government vs rebels with caught-in-the-middle civilians",
        "factions": [
            _make_faction("gov", "Government Forces", "#00f0ff", "government", 0.8, 0.7),
            _make_faction("reb", "Rebel Movement", "#ff2a6d", "rebel", 0.4, 0.2, 0.8),
            _make_faction("civ", "Civilian Population", "#fcee0a", "civilian", 0.05, 0.3, 0.5),
        ],
        "initial_relations": [
            ("gov", "reb", Relation.WAR),
            ("gov", "civ", Relation.FRIENDLY),
            ("reb", "civ", Relation.SUSPICIOUS),
        ],
    },
    "peacekeeping": {
        "description": "UN peacekeepers, government, rebels, and civilians",
        "factions": [
            _make_faction("un", "UN Peacekeepers", "#0088ff", "peacekeeping", 0.6, 0.8, 0.9),
            _make_faction("gov", "Government Forces", "#00f0ff", "government", 0.7, 0.6),
            _make_faction("reb", "Rebel Forces", "#ff2a6d", "rebel", 0.4, 0.2, 0.8),
            _make_faction("civ", "Civilian Population", "#fcee0a", "civilian", 0.05, 0.3, 0.5),
        ],
        "initial_relations": [
            ("un", "gov", Relation.FRIENDLY),
            ("un", "reb", Relation.NEUTRAL),
            ("un", "civ", Relation.FRIENDLY),
            ("gov", "reb", Relation.WAR),
            ("gov", "civ", Relation.FRIENDLY),
            ("reb", "civ", Relation.SUSPICIOUS),
        ],
    },
}


def load_preset(name: str) -> DiplomacyEngine:
    """Create a DiplomacyEngine from a named preset.

    Raises ``KeyError`` if *name* is not in :data:`FACTION_PRESETS`.
    """
    preset = FACTION_PRESETS[name]
    engine = DiplomacyEngine()
    for f in preset["factions"]:
        engine.add_faction(f)
    for a, b, rel in preset.get("initial_relations", []):
        engine.set_relation(a, b, rel)
    return engine

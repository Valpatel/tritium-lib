"""Medical, casualty, triage, and evacuation system for the Tritium sim engine.

Simulates injuries with body-part specificity, bleeding, triage classification,
medic treatment, and casualty evacuation.  Integrates with the unit system for
morale/mobility/accuracy penalties.

Copyright 2026 Valpatel Software LLC -- AGPL-3.0
"""

from __future__ import annotations

import enum
import random
import uuid
from dataclasses import dataclass, field

from tritium_lib.sim_engine.ai.steering import Vec2


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InjuryType(enum.Enum):
    GUNSHOT = "gunshot"
    SHRAPNEL = "shrapnel"
    BURN = "burn"
    BLAST = "blast"
    CRUSH = "crush"
    LACERATION = "laceration"
    CONCUSSION = "concussion"


class InjurySeverity(enum.Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    FATAL = "fatal"


class TriageCategory(enum.Enum):
    IMMEDIATE = "immediate"    # red
    DELAYED = "delayed"        # yellow
    MINIMAL = "minimal"        # green
    EXPECTANT = "expectant"    # black


TRIAGE_COLORS: dict[TriageCategory, str] = {
    TriageCategory.IMMEDIATE: "#ff0000",
    TriageCategory.DELAYED: "#ffff00",
    TriageCategory.MINIMAL: "#00ff00",
    TriageCategory.EXPECTANT: "#000000",
}


BODY_PARTS = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")


# ---------------------------------------------------------------------------
# Injury tables — probability distributions per injury type
# ---------------------------------------------------------------------------

# Maps InjuryType -> list of (body_part, probability)
INJURY_TABLES: dict[InjuryType, list[tuple[str, float]]] = {
    InjuryType.GUNSHOT: [
        ("torso", 0.40),
        ("left_arm", 0.10),
        ("right_arm", 0.10),
        ("left_leg", 0.10),
        ("right_leg", 0.10),
        ("head", 0.10),
    ],
    InjuryType.SHRAPNEL: [
        ("head", 1 / 6),
        ("torso", 1 / 6),
        ("left_arm", 1 / 6),
        ("right_arm", 1 / 6),
        ("left_leg", 1 / 6),
        ("right_leg", 1 / 6),
    ],
    InjuryType.BURN: [
        ("torso", 0.30),
        ("left_arm", 0.15),
        ("right_arm", 0.15),
        ("left_leg", 0.10),
        ("right_leg", 0.10),
        ("head", 0.20),
    ],
    InjuryType.BLAST: [
        # Blast primarily causes concussion + secondary shrapnel
        ("head", 0.40),
        ("torso", 0.30),
        ("left_arm", 0.05),
        ("right_arm", 0.05),
        ("left_leg", 0.10),
        ("right_leg", 0.10),
    ],
    InjuryType.CRUSH: [
        ("torso", 0.30),
        ("left_leg", 0.20),
        ("right_leg", 0.20),
        ("left_arm", 0.10),
        ("right_arm", 0.10),
        ("head", 0.10),
    ],
    InjuryType.LACERATION: [
        ("left_arm", 0.20),
        ("right_arm", 0.20),
        ("torso", 0.20),
        ("left_leg", 0.15),
        ("right_leg", 0.15),
        ("head", 0.10),
    ],
    InjuryType.CONCUSSION: [
        ("head", 1.0),
    ],
}


# Severity weights by body part (head injuries trend more severe)
_SEVERITY_WEIGHTS: dict[str, list[tuple[InjurySeverity, float]]] = {
    "head": [
        (InjurySeverity.MINOR, 0.05),
        (InjurySeverity.MODERATE, 0.15),
        (InjurySeverity.SEVERE, 0.30),
        (InjurySeverity.CRITICAL, 0.35),
        (InjurySeverity.FATAL, 0.15),
    ],
    "torso": [
        (InjurySeverity.MINOR, 0.10),
        (InjurySeverity.MODERATE, 0.25),
        (InjurySeverity.SEVERE, 0.35),
        (InjurySeverity.CRITICAL, 0.20),
        (InjurySeverity.FATAL, 0.10),
    ],
    "left_arm": [
        (InjurySeverity.MINOR, 0.30),
        (InjurySeverity.MODERATE, 0.35),
        (InjurySeverity.SEVERE, 0.25),
        (InjurySeverity.CRITICAL, 0.08),
        (InjurySeverity.FATAL, 0.02),
    ],
    "right_arm": [
        (InjurySeverity.MINOR, 0.30),
        (InjurySeverity.MODERATE, 0.35),
        (InjurySeverity.SEVERE, 0.25),
        (InjurySeverity.CRITICAL, 0.08),
        (InjurySeverity.FATAL, 0.02),
    ],
    "left_leg": [
        (InjurySeverity.MINOR, 0.25),
        (InjurySeverity.MODERATE, 0.35),
        (InjurySeverity.SEVERE, 0.25),
        (InjurySeverity.CRITICAL, 0.10),
        (InjurySeverity.FATAL, 0.05),
    ],
    "right_leg": [
        (InjurySeverity.MINOR, 0.25),
        (InjurySeverity.MODERATE, 0.35),
        (InjurySeverity.SEVERE, 0.25),
        (InjurySeverity.CRITICAL, 0.10),
        (InjurySeverity.FATAL, 0.05),
    ],
}

# Bleed rate (hp/s) by severity
_BLEED_RATES: dict[InjurySeverity, float] = {
    InjurySeverity.MINOR: 0.005,
    InjurySeverity.MODERATE: 0.02,
    InjurySeverity.SEVERE: 0.05,
    InjurySeverity.CRITICAL: 0.10,
    InjurySeverity.FATAL: 0.20,
}

# Pain by severity (0-1)
_PAIN_VALUES: dict[InjurySeverity, float] = {
    InjurySeverity.MINOR: 0.1,
    InjurySeverity.MODERATE: 0.3,
    InjurySeverity.SEVERE: 0.5,
    InjurySeverity.CRITICAL: 0.8,
    InjurySeverity.FATAL: 1.0,
}

# Treatment time multipliers (seconds to treat one injury of given severity)
_TREATMENT_TIMES: dict[InjurySeverity, float] = {
    InjurySeverity.MINOR: 5.0,
    InjurySeverity.MODERATE: 15.0,
    InjurySeverity.SEVERE: 30.0,
    InjurySeverity.CRITICAL: 60.0,
    InjurySeverity.FATAL: 120.0,
}

# Injury type modifiers for bleed rate
_TYPE_BLEED_MODIFIER: dict[InjuryType, float] = {
    InjuryType.GUNSHOT: 1.5,
    InjuryType.SHRAPNEL: 1.2,
    InjuryType.BURN: 0.3,       # burns bleed less
    InjuryType.BLAST: 0.5,
    InjuryType.CRUSH: 0.8,
    InjuryType.LACERATION: 1.0,
    InjuryType.CONCUSSION: 0.0, # concussions don't bleed
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Injury:
    """A single injury on a unit."""

    injury_id: str
    injury_type: InjuryType
    severity: InjurySeverity
    body_part: str
    bleed_rate: float          # hp/s blood loss
    pain: float                # 0-1, affects morale
    mobility_penalty: float    # 0-1, leg injuries slow movement
    accuracy_penalty: float    # 0-1, arm injuries hurt aim
    treated: bool = False
    time_since: float = 0.0    # seconds since injury

    def is_limb_injury(self) -> bool:
        """Return True if this injury affects a limb."""
        return self.body_part in ("left_arm", "right_arm", "left_leg", "right_leg")


@dataclass
class CasualtyState:
    """Medical state for a single unit."""

    unit_id: str
    injuries: list[Injury] = field(default_factory=list)
    blood_level: float = 1.0          # 0=dead, <0.3=unconscious, <0.5=critical
    consciousness: bool = True
    triage: TriageCategory = TriageCategory.MINIMAL
    being_treated_by: str | None = None
    evacuation_status: str = "none"    # none, requested, in_transit, evacuated
    position: Vec2 = (0.0, 0.0)

    @property
    def is_dead(self) -> bool:
        return self.blood_level <= 0.0

    @property
    def total_bleed_rate(self) -> float:
        """Sum of active (untreated) bleed rates."""
        return sum(i.bleed_rate for i in self.injuries if not i.treated)

    @property
    def total_pain(self) -> float:
        """Aggregate pain level, capped at 1.0."""
        return min(1.0, sum(i.pain for i in self.injuries))

    @property
    def total_mobility_penalty(self) -> float:
        """Aggregate mobility penalty, capped at 1.0."""
        return min(1.0, sum(i.mobility_penalty for i in self.injuries if not i.treated))

    @property
    def total_accuracy_penalty(self) -> float:
        """Aggregate accuracy penalty, capped at 1.0."""
        return min(1.0, sum(i.accuracy_penalty for i in self.injuries if not i.treated))

    @property
    def has_fatal_injury(self) -> bool:
        return any(i.severity == InjurySeverity.FATAL for i in self.injuries)

    @property
    def worst_severity(self) -> InjurySeverity | None:
        if not self.injuries:
            return None
        order = list(InjurySeverity)
        return max(self.injuries, key=lambda i: order.index(i.severity)).severity


# ---------------------------------------------------------------------------
# Evac request
# ---------------------------------------------------------------------------

@dataclass
class EvacRequest:
    """A request to evacuate a casualty to an evac point."""

    unit_id: str
    evac_point: Vec2
    priority: TriageCategory
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# MedicalEngine
# ---------------------------------------------------------------------------

class MedicalEngine:
    """Manages injuries, triage, treatment, and casualty evacuation."""

    def __init__(self) -> None:
        self.casualties: dict[str, CasualtyState] = {}
        self.medics: dict[str, str] = {}  # medic_id -> patient_id
        self._treatment_progress: dict[str, float] = {}  # medic_id -> progress 0-1
        self._evac_requests: list[EvacRequest] = []

    # -- helpers -----------------------------------------------------------

    def _get_or_create(self, unit_id: str) -> CasualtyState:
        """Retrieve or lazily create a CasualtyState for *unit_id*."""
        if unit_id not in self.casualties:
            self.casualties[unit_id] = CasualtyState(unit_id=unit_id)
        return self.casualties[unit_id]

    @staticmethod
    def _pick_weighted(
        choices: list[tuple[str, float]],
        rng: random.Random | None = None,
    ) -> str:
        """Weighted random pick from (value, weight) pairs."""
        r = rng or random.Random()
        roll = r.random()
        cumulative = 0.0
        for value, weight in choices:
            cumulative += weight
            if roll <= cumulative:
                return value
        return choices[-1][0]

    @staticmethod
    def _pick_severity_weighted(
        choices: list[tuple[InjurySeverity, float]],
        rng: random.Random | None = None,
    ) -> InjurySeverity:
        """Weighted random pick for severity."""
        r = rng or random.Random()
        roll = r.random()
        cumulative = 0.0
        for sev, weight in choices:
            cumulative += weight
            if roll <= cumulative:
                return sev
        return choices[-1][0]

    # -- core API ----------------------------------------------------------

    def inflict_injury(
        self,
        unit_id: str,
        injury_type: InjuryType,
        body_part: str | None = None,
        severity: InjurySeverity | None = None,
        rng: random.Random | None = None,
    ) -> Injury:
        """Inflict an injury on a unit.

        If *body_part* is None, pick one from the injury probability table.
        If *severity* is None, auto-calculate from type + body part.
        Returns the created Injury.
        """
        r = rng or random.Random()
        cs = self._get_or_create(unit_id)

        # Pick body part from injury table if not specified
        if body_part is None:
            table = INJURY_TABLES.get(injury_type, INJURY_TABLES[InjuryType.SHRAPNEL])
            body_part = self._pick_weighted(table, r)

        # Validate body part
        if body_part not in BODY_PARTS:
            body_part = "torso"

        # Auto-determine severity
        if severity is None:
            weights = _SEVERITY_WEIGHTS.get(body_part, _SEVERITY_WEIGHTS["torso"])
            severity = self._pick_severity_weighted(weights, r)

        # Calculate bleed rate
        base_bleed = _BLEED_RATES[severity]
        type_mod = _TYPE_BLEED_MODIFIER.get(injury_type, 1.0)
        bleed_rate = base_bleed * type_mod

        # Calculate pain
        pain = _PAIN_VALUES[severity]

        # Calculate mobility penalty (leg injuries)
        mobility_penalty = 0.0
        if body_part in ("left_leg", "right_leg"):
            mobility_penalty = {
                InjurySeverity.MINOR: 0.1,
                InjurySeverity.MODERATE: 0.25,
                InjurySeverity.SEVERE: 0.5,
                InjurySeverity.CRITICAL: 0.8,
                InjurySeverity.FATAL: 1.0,
            }[severity]

        # Calculate accuracy penalty (arm injuries)
        accuracy_penalty = 0.0
        if body_part in ("left_arm", "right_arm"):
            accuracy_penalty = {
                InjurySeverity.MINOR: 0.1,
                InjurySeverity.MODERATE: 0.2,
                InjurySeverity.SEVERE: 0.4,
                InjurySeverity.CRITICAL: 0.7,
                InjurySeverity.FATAL: 1.0,
            }[severity]
        # Head injuries also affect accuracy (disorientation)
        elif body_part == "head":
            accuracy_penalty = {
                InjurySeverity.MINOR: 0.05,
                InjurySeverity.MODERATE: 0.15,
                InjurySeverity.SEVERE: 0.3,
                InjurySeverity.CRITICAL: 0.6,
                InjurySeverity.FATAL: 1.0,
            }[severity]

        injury = Injury(
            injury_id=uuid.uuid4().hex[:8],
            injury_type=injury_type,
            severity=severity,
            body_part=body_part,
            bleed_rate=bleed_rate,
            pain=pain,
            mobility_penalty=mobility_penalty,
            accuracy_penalty=accuracy_penalty,
        )

        cs.injuries.append(injury)
        cs.triage = self.triage(unit_id)
        return injury

    def inflict_blast(
        self,
        unit_id: str,
        distance_from_center: float,
        blast_radius: float,
        rng: random.Random | None = None,
    ) -> list[Injury]:
        """Inflict blast injuries — concussion + possible shrapnel.

        Closer to the blast center = more severe.
        """
        r = rng or random.Random()
        injuries: list[Injury] = []

        if distance_from_center > blast_radius:
            return injuries

        ratio = distance_from_center / max(blast_radius, 0.01)

        # Always get a concussion from a blast
        if ratio < 0.3:
            conc_sev = InjurySeverity.CRITICAL
        elif ratio < 0.6:
            conc_sev = InjurySeverity.SEVERE
        elif ratio < 0.8:
            conc_sev = InjurySeverity.MODERATE
        else:
            conc_sev = InjurySeverity.MINOR

        injuries.append(self.inflict_injury(
            unit_id, InjuryType.CONCUSSION, "head", conc_sev, r,
        ))

        # 60% chance of shrapnel at close range, scaling down
        shrapnel_chance = max(0.0, 0.6 * (1.0 - ratio))
        if r.random() < shrapnel_chance:
            injuries.append(self.inflict_injury(
                unit_id, InjuryType.SHRAPNEL, rng=r,
            ))

        return injuries

    def inflict_burn(
        self,
        unit_id: str,
        distance_from_fire: float,
        fire_radius: float,
        rng: random.Random | None = None,
    ) -> Injury | None:
        """Inflict burn injuries based on distance from fire source."""
        r = rng or random.Random()

        if distance_from_fire > fire_radius:
            return None

        ratio = distance_from_fire / max(fire_radius, 0.01)

        if ratio < 0.2:
            sev = InjurySeverity.CRITICAL
        elif ratio < 0.4:
            sev = InjurySeverity.SEVERE
        elif ratio < 0.7:
            sev = InjurySeverity.MODERATE
        else:
            sev = InjurySeverity.MINOR

        return self.inflict_injury(unit_id, InjuryType.BURN, severity=sev, rng=r)

    # -- triage ------------------------------------------------------------

    def triage(self, unit_id: str) -> TriageCategory:
        """Classify a casualty into a triage category.

        - EXPECTANT: fatal injuries or blood level near zero — conserve resources.
        - IMMEDIATE: severe/critical bleeding, still treatable.
        - DELAYED: moderate injuries, stable for now.
        - MINIMAL: walking wounded.
        """
        cs = self.casualties.get(unit_id)
        if cs is None:
            return TriageCategory.MINIMAL

        # Dead or dying — expectant
        if cs.is_dead or cs.blood_level < 0.1:
            return TriageCategory.EXPECTANT

        if cs.has_fatal_injury:
            return TriageCategory.EXPECTANT

        worst = cs.worst_severity
        if worst is None:
            return TriageCategory.MINIMAL

        # Critical or heavy bleeding
        if worst in (InjurySeverity.CRITICAL,) or cs.blood_level < 0.4:
            return TriageCategory.IMMEDIATE

        if worst == InjurySeverity.SEVERE or cs.blood_level < 0.6:
            return TriageCategory.IMMEDIATE

        if worst == InjurySeverity.MODERATE:
            return TriageCategory.DELAYED

        return TriageCategory.MINIMAL

    # -- treatment ---------------------------------------------------------

    def assign_medic(self, medic_id: str, patient_id: str) -> bool:
        """Assign a medic to treat a patient.

        Returns False if the patient is dead or already being treated.
        """
        cs = self.casualties.get(patient_id)
        if cs is None or cs.is_dead:
            return False

        # Release previous patient if any
        if medic_id in self.medics:
            old_patient = self.medics[medic_id]
            old_cs = self.casualties.get(old_patient)
            if old_cs and old_cs.being_treated_by == medic_id:
                old_cs.being_treated_by = None

        self.medics[medic_id] = patient_id
        cs.being_treated_by = medic_id
        self._treatment_progress[medic_id] = 0.0
        return True

    def release_medic(self, medic_id: str) -> None:
        """Release a medic from treating their current patient."""
        if medic_id in self.medics:
            patient_id = self.medics[medic_id]
            cs = self.casualties.get(patient_id)
            if cs and cs.being_treated_by == medic_id:
                cs.being_treated_by = None
            del self.medics[medic_id]
        self._treatment_progress.pop(medic_id, None)

    def treat(self, medic_id: str, patient_id: str, dt: float) -> dict:
        """Advance treatment for *dt* seconds.

        Returns a dict with treatment progress info.
        """
        cs = self.casualties.get(patient_id)
        if cs is None or cs.is_dead:
            return {"status": "no_patient", "progress": 0.0}

        # Auto-assign if not already
        if medic_id not in self.medics or self.medics[medic_id] != patient_id:
            if not self.assign_medic(medic_id, patient_id):
                return {"status": "cannot_treat", "progress": 0.0}

        # Find the most urgent untreated injury
        untreated = [i for i in cs.injuries if not i.treated]
        if not untreated:
            self.release_medic(medic_id)
            return {"status": "all_treated", "progress": 1.0}

        # Treat the most severe untreated injury first
        order = list(InjurySeverity)
        untreated.sort(key=lambda i: order.index(i.severity), reverse=True)
        target_injury = untreated[0]

        # Calculate treatment progress
        treatment_time = _TREATMENT_TIMES[target_injury.severity]
        progress = self._treatment_progress.get(medic_id, 0.0)
        progress += dt / treatment_time
        self._treatment_progress[medic_id] = progress

        result: dict = {
            "status": "treating",
            "injury_id": target_injury.injury_id,
            "injury_type": target_injury.injury_type.value,
            "severity": target_injury.severity.value,
            "progress": min(1.0, progress),
        }

        if progress >= 1.0:
            # Treatment complete for this injury
            target_injury.treated = True
            target_injury.bleed_rate = 0.0
            target_injury.pain *= 0.3  # pain reduced but not gone
            target_injury.mobility_penalty *= 0.5
            target_injury.accuracy_penalty *= 0.5
            self._treatment_progress[medic_id] = 0.0
            result["status"] = "injury_treated"

            # Check if there are more to treat
            remaining = [i for i in cs.injuries if not i.treated]
            result["remaining"] = len(remaining)
            if not remaining:
                self.release_medic(medic_id)
                result["status"] = "all_treated"

            # Re-triage
            cs.triage = self.triage(patient_id)

        return result

    # -- evacuation --------------------------------------------------------

    def request_evac(
        self,
        unit_id: str,
        evac_point: Vec2,
        timestamp: float = 0.0,
    ) -> EvacRequest | None:
        """Request casualty evacuation to *evac_point*.

        Returns the EvacRequest, or None if the unit has no injuries.
        """
        cs = self.casualties.get(unit_id)
        if cs is None:
            return None

        cs.evacuation_status = "requested"
        req = EvacRequest(
            unit_id=unit_id,
            evac_point=evac_point,
            priority=cs.triage,
            timestamp=timestamp,
        )
        self._evac_requests.append(req)
        return req

    def update_evac_status(self, unit_id: str, status: str) -> None:
        """Update evacuation status (none, requested, in_transit, evacuated)."""
        cs = self.casualties.get(unit_id)
        if cs:
            cs.evacuation_status = status

    @property
    def evac_requests(self) -> list[EvacRequest]:
        """All active evacuation requests."""
        return list(self._evac_requests)

    def clear_evac(self, unit_id: str) -> None:
        """Remove evacuation request for a unit."""
        self._evac_requests = [r for r in self._evac_requests if r.unit_id != unit_id]
        cs = self.casualties.get(unit_id)
        if cs:
            cs.evacuation_status = "evacuated"

    # -- tick --------------------------------------------------------------

    def tick(self, dt: float) -> list[dict]:
        """Advance the medical simulation by *dt* seconds.

        - Bleeding reduces blood_level.
        - Low blood causes unconsciousness, then death.
        - Untreated injuries worsen over time.

        Returns a list of event dicts (death, unconscious, worsened).
        """
        events: list[dict] = []

        for unit_id, cs in list(self.casualties.items()):
            if cs.is_dead:
                continue

            # Advance time_since on all injuries
            for inj in cs.injuries:
                inj.time_since += dt

            # Bleeding
            bleed = cs.total_bleed_rate * dt
            if bleed > 0:
                cs.blood_level = max(0.0, cs.blood_level - bleed)

            # Untreated injuries worsen over time
            for inj in cs.injuries:
                if inj.treated:
                    continue
                if inj.severity == InjurySeverity.FATAL:
                    continue
                # Small chance of worsening per tick (1% per second for severe+)
                if inj.time_since > 60.0 and inj.severity in (
                    InjurySeverity.SEVERE,
                    InjurySeverity.CRITICAL,
                ):
                    worsen_rate = 0.001 * dt  # ~0.1% per second
                    # Deterministic worsening: bleed rate increases
                    inj.bleed_rate *= (1.0 + worsen_rate)

            # Consciousness check
            was_conscious = cs.consciousness
            if cs.blood_level < 0.3:
                cs.consciousness = False
                if was_conscious:
                    events.append({
                        "type": "unconscious",
                        "unit_id": unit_id,
                        "blood_level": cs.blood_level,
                    })

            # Death check
            if cs.blood_level <= 0.0:
                cs.blood_level = 0.0
                cs.consciousness = False
                events.append({
                    "type": "death",
                    "unit_id": unit_id,
                })

            # Update triage
            cs.triage = self.triage(unit_id)

        return events

    # -- reporting ---------------------------------------------------------

    def get_casualty_report(self) -> dict:
        """Summary of all casualties grouped by triage category."""
        by_triage: dict[str, list[str]] = {
            cat.value: [] for cat in TriageCategory
        }
        total_injuries = 0
        total_dead = 0
        total_unconscious = 0

        for unit_id, cs in self.casualties.items():
            by_triage[cs.triage.value].append(unit_id)
            total_injuries += len(cs.injuries)
            if cs.is_dead:
                total_dead += 1
            elif not cs.consciousness:
                total_unconscious += 1

        return {
            "total_casualties": len(self.casualties),
            "total_injuries": total_injuries,
            "total_dead": total_dead,
            "total_unconscious": total_unconscious,
            "by_triage": by_triage,
            "evac_pending": len([
                r for r in self._evac_requests
                if self.casualties.get(r.unit_id, CasualtyState(unit_id=""))
                .evacuation_status == "requested"
            ]),
        }

    def get_unit_injuries(self, unit_id: str) -> list[dict]:
        """Detailed injury list for one unit."""
        cs = self.casualties.get(unit_id)
        if cs is None:
            return []
        return [
            {
                "injury_id": inj.injury_id,
                "type": inj.injury_type.value,
                "severity": inj.severity.value,
                "body_part": inj.body_part,
                "bleed_rate": inj.bleed_rate,
                "pain": inj.pain,
                "mobility_penalty": inj.mobility_penalty,
                "accuracy_penalty": inj.accuracy_penalty,
                "treated": inj.treated,
                "time_since": inj.time_since,
            }
            for inj in cs.injuries
        ]

    # -- Three.js rendering ------------------------------------------------

    def to_three_js(self) -> dict:
        """Export medical state for Three.js visualization."""
        casualties_out: list[dict] = []
        medics_out: list[dict] = []
        evac_out: list[dict] = []

        for unit_id, cs in self.casualties.items():
            if not cs.injuries:
                continue
            casualties_out.append({
                "id": unit_id,
                "x": cs.position[0],
                "y": cs.position[1],
                "triage": cs.triage.value,
                "color": TRIAGE_COLORS[cs.triage],
                "blood_level": round(cs.blood_level, 2),
                "conscious": cs.consciousness,
                "injury_count": len(cs.injuries),
                "treated_count": sum(1 for i in cs.injuries if i.treated),
            })

        for medic_id, patient_id in self.medics.items():
            medics_out.append({
                "id": medic_id,
                "treating": patient_id,
                "progress": round(self._treatment_progress.get(medic_id, 0.0), 2),
            })

        for req in self._evac_requests:
            cs = self.casualties.get(req.unit_id)
            if cs and cs.evacuation_status == "requested":
                evac_out.append({
                    "id": req.unit_id,
                    "x": req.evac_point[0],
                    "y": req.evac_point[1],
                    "priority": req.priority.value,
                })

        return {
            "casualties": casualties_out,
            "medics": medics_out,
            "evac_requests": evac_out,
        }

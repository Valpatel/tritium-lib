# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""KETTLE ON REAL TERRAIN — the terrain-aware slot-validator A/B proof.

The costmap lane found that when a bloc is kettled against a building, the
police cordon slots that land inside the wall leave officers BEELINING into the
structure and stalling — the cordon never CLOSES (``kettle_formed`` never
fires; the mean police->bloc-centroid distance stays large).

``PoliceTacticsController`` exposes an injected ``slot_validator`` seam.  This
drives the SAME controller over a real :class:`Costmap` with a wall clipping
the cordon ring, moving officers faithfully (a straight step that would cross a
lethal cell is blocked, mirroring the engine's collision check), and proves the
A/B:

  * validator OFF  -> officers jam on the wall, ``kettle_formed`` never fires,
                      the cordon stays loose.
  * validator ON   -> the wall-side slots are nudged to reachable clear cells,
                      officers close the ring, ``kettle_formed`` FIRES, and the
                      mean police->centroid distance TIGHTENS.

Pure lib: PoliceTacticsController + ``planning.validate_slot`` + ``Costmap`` —
no engine, deterministic (no RNG).
"""
from __future__ import annotations

import math

from tritium_lib.planning import validate_slot
from tritium_lib.planning.costmap import Costmap
from tritium_lib.sim_engine.game.riot_police import PoliceTacticsController


class _Bus:
    def __init__(self) -> None:
        self.beats: list[str] = []

    def publish(self, topic: str, data) -> None:
        if topic == "crowd_event" and isinstance(data, dict):
            self.beats.append(data.get("beat"))

    def subscribe(self, *a, **k):
        return None


class _GameMode:
    game_mode_type = "civil_unrest"
    de_escalation_score = 0
    de_escalation_target = 1_000_000  # never "win"; keep the cordon running
    arrest_count = 0
    rout_count = 0


class _Unit:
    """Minimal SimulationTarget stand-in the controller reads/writes."""

    def __init__(self, tid, x, y, *, police, health=100.0):
        self.target_id = tid
        self.position = [float(x), float(y)]
        self.waypoints: list[tuple[float, float]] = []
        self.alliance = "friendly" if police else "hostile"
        self.asset_type = "police" if police else "person"
        self.crowd_role = None if police else "rioter"
        self.status = "active"
        self.health = health
        self.last_fired = -1e9
        self.is_combatant = not police


def _wall_costmap() -> Costmap:
    """80x80m grid @ 1m, with an EAST wall clipping the cordon's east arc.

    Lethal block x in [7, 22], y in [-16, 16] (grid rows/cols offset by +40).
    The bloc sits at the origin; a ~12 m cordon ring's eastern slots fall inside
    this wall.
    """
    n = 80
    grid = [[1.0] * n for _ in range(n)]
    for gy in range(n):
        for gx in range(n):
            wx = gx - 40
            wy = gy - 40
            if 7 <= wx <= 22 and -16 <= wy <= 16:
                grid[gy][gx] = Costmap.LETHAL
    return Costmap(origin_x=-40.0, origin_y=-40.0, resolution=1.0,
                   width=n, height=n, grid=grid)


def _run(with_validator: bool) -> dict:
    cm = _wall_costmap()
    bus = _Bus()
    pc = PoliceTacticsController(bus, _GameMode())
    if with_validator:
        pc.set_slot_validator(
            lambda frm, slot: validate_slot(cm, frm, slot, clearance_m=0.0)
        )

    # A tight rioter bloc at the origin (high health so nobody is arrested /
    # routed during the proof window — the cluster stays put to kettle).
    rioters = [
        _Unit(f"riot_{i}", math.cos(i) * 2.0, math.sin(i) * 2.0,
              police=False, health=100.0)
        for i in range(6)
    ]
    # Eight officers massed to the SOUTH (auto gap opens north, away from them).
    officers = [
        _Unit(f"cop_{i}", -14.0 + i * 4.0, -30.0, police=True)
        for i in range(8)
    ]
    targets = {u.target_id: u for u in rioters + officers}

    pc.command_tactic("kettle")  # operator kettle of the local cluster

    dt = 0.1
    speed = 3.0
    kettle_formed = False
    min_mean_d = 1e9
    for _ in range(600):  # 60 s
        pc.tick(dt, targets, "civil_unrest")
        # Faithful officer movement: step toward the current waypoint, but a
        # step that would cross a lethal cell is BLOCKED (engine collision).
        for o in officers:
            if not o.waypoints:
                continue
            wx, wy = o.waypoints[0]
            dx, dy = wx - o.position[0], wy - o.position[1]
            d = math.hypot(dx, dy)
            if d < 1e-9:
                continue
            step = min(speed * dt, d)
            nx = o.position[0] + dx / d * step
            ny = o.position[1] + dy / d * step
            cell = cm.world_to_grid(nx, ny)
            if cell is not None and cm.is_lethal(*cell):
                continue  # wall — officer stalls (beelined into the building)
            o.position[0], o.position[1] = nx, ny
        if "kettle_formed" in bus.beats:
            kettle_formed = True
        # Mean police -> bloc-centroid distance (cordon tightness).
        cx = sum(r.position[0] for r in rioters) / len(rioters)
        cy = sum(r.position[1] for r in rioters) / len(rioters)
        mean_d = sum(math.hypot(o.position[0] - cx, o.position[1] - cy)
                     for o in officers) / len(officers)
        min_mean_d = min(min_mean_d, mean_d)

    # Cordon adherence: officers that actually reached their commanded slot
    # (within the arrival tolerance).  Officers jammed on the wall never do.
    arrived = 0
    for o in officers:
        if o.waypoints:
            wx, wy = o.waypoints[0]
            if math.hypot(o.position[0] - wx, o.position[1] - wy) <= 3.0:
                arrived += 1
    officers_in_wall = sum(
        1 for o in officers
        if (c := cm.world_to_grid(*o.position)) is not None and cm.is_lethal(*c)
    )
    return {
        "kettle_formed": kettle_formed,
        "min_mean_d": min_mean_d,
        "arrived": arrived,
        "n_officers": len(officers),
        "officers_in_wall": officers_in_wall,
    }


def test_validator_closes_the_cordon_on_terrain():
    off = _run(with_validator=False)
    on = _run(with_validator=True)

    # THE PROOF: only with the validator does the cordon actually CLOSE.
    assert on["kettle_formed"], (
        "cordon never closed WITH the validator (kettle_formed did not fire)"
    )
    assert not off["kettle_formed"], (
        "control regressed: the cordon closed WITHOUT the validator — the wall "
        "no longer blocks the beeline, so this A/B proves nothing"
    )

    # Cordon ADHERENCE: with the validator a cordon-forming majority reaches its
    # (nudged, reachable) slot even as the ring keeps tightening; without it the
    # wall-side officers stall so fewer arrive.
    assert on["arrived"] >= math.ceil(0.75 * on["n_officers"]), (
        f"only {on['arrived']}/{on['n_officers']} officers reached their slot "
        f"WITH the validator (need a >=75% cordon)"
    )
    assert on["arrived"] > off["arrived"], (
        f"validator did not improve cordon adherence: ON {on['arrived']} vs "
        f"OFF {off['arrived']} arrived (the wall must block someone off)"
    )

    # The cordon TIGHTENS: mean police->centroid distance shrinks (the wall-side
    # officers close onto the reachable ring instead of stalling off it).
    assert on["min_mean_d"] < off["min_mean_d"], (
        f"cordon did not tighten: ON {on['min_mean_d']:.1f}m vs "
        f"OFF {off['min_mean_d']:.1f}m"
    )

    # No officer is left standing inside a building with the validator on.
    assert on["officers_in_wall"] == 0, (
        f"{on['officers_in_wall']} officer(s) ended inside a wall WITH the "
        f"validator — the nudge did not keep them on valid cells"
    )


def test_open_ground_is_identity():
    """With no wall the validator must be a pure identity (no cordon change)."""
    n = 40
    grid = [[1.0] * n for _ in range(n)]
    cm = Costmap(origin_x=-20.0, origin_y=-20.0, resolution=1.0,
                 width=n, height=n, grid=grid)
    frm = (0.0, -15.0)
    for slot in [(5.0, 5.0), (-8.0, 3.0), (0.0, 10.0), (7.3, -2.1)]:
        assert validate_slot(cm, frm, slot, 0.0) == slot

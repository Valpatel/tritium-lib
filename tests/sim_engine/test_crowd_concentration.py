# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Non-gameable spatial-concentration measurement harness for the riot crowd.

MEASURE-FIRST harness (no fixes). It measures where bodies ACTUALLY ARE on the
realized member positions — never mood labels — so the numbers can't be gamed
by relabelling moods. Three scenarios:

  1) seeded_riot   — _build_riot(bounds, seed=42): the GOOD legibility baseline.
  2) operator_spawn — faithful reproduction of WorldBuilder.spawn_crowd: ONE
                       CrowdSimulator, ONE spawn_crowd at centre, NO anchor, NO
                       seed/rng. The SUSPECT central pile.
  3) central_surge  — from seeded_riot steady state, inject ONE high-intensity
                       CrowdEvent at the centre, peak then decay.

Determinism: every path through the crowd code uses the module-global ``random``
for per-tick jitter (and the operator path uses it for spawn placement too), so
we seed ``random.seed(...)`` at the start of each scenario. _build_riot also
takes an explicit ``seed`` for its placement RNG and objective RNG.

Run standalone (prints actual numbers):
    /home/scubasonar/Code/tritium/tritium-sc/.venv/bin/python \
        tests/sim_engine/test_crowd_concentration.py

Or as pytest:
    pytest tests/sim_engine/test_crowd_concentration.py -s
"""

from __future__ import annotations

import math
import random

from tritium_lib.sim_engine.crowd import (
    CrowdSimulator,
    CrowdEvent,
    CrowdMood,
    _build_riot,
    _build_stampede,
    _build_standoff,
)


# ---------------------------------------------------------------------------
# Metrics — all computed from realized member positions only.
# ---------------------------------------------------------------------------

def _positions(sim: CrowdSimulator) -> list[tuple[float, float]]:
    return [tuple(m.position) for m in sim.members]


def _centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(pts)
    if n == 0:
        return (0.0, 0.0)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def r50_m(pts: list[tuple[float, float]]) -> float:
    """Radius from the member centroid containing 50% of members.
    Larger = more spread."""
    n = len(pts)
    if n == 0:
        return 0.0
    cx, cy = _centroid(pts)
    dists = sorted(math.hypot(p[0] - cx, p[1] - cy) for p in pts)
    # median distance (the radius holding the closer 50%)
    return dists[n // 2]


def densest_cell_frac(pts: list[tuple[float, float]], cell: float = 10.0) -> float:
    """Fraction of members in the single most-occupied 10 m grid cell.
    Smaller = more spread."""
    n = len(pts)
    if n == 0:
        return 0.0
    cells: dict[tuple[int, int], int] = {}
    for x, y in pts:
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        cells[key] = cells.get(key, 0) + 1
    return max(cells.values()) / n


def peak_local_density(pts: list[tuple[float, float]], radius: float = 5.0) -> float:
    """Max over members of neighbour count within 5 m (excluding self).
    Brute force O(n^2) — independent of the sim's own grid, so non-gameable."""
    n = len(pts)
    if n == 0:
        return 0.0
    r2 = radius * radius
    best = 0
    for i in range(n):
        xi, yi = pts[i]
        c = 0
        for j in range(n):
            if i == j:
                continue
            dx = pts[j][0] - xi
            dy = pts[j][1] - yi
            if dx * dx + dy * dy <= r2:
                c += 1
        if c > best:
            best = c
    return float(best)


def sector_occupancy_frac(
    pts: list[tuple[float, float]],
    bounds: tuple[float, float, float, float],
) -> float:
    """Fraction of a 3x3 bounds grid whose cells contain >=1 member (/9).
    Larger = uses more of the city."""
    n = len(pts)
    if n == 0:
        return 0.0
    x0, y0, x1, y1 = bounds
    w = (x1 - x0) / 3.0
    h = (y1 - y0) / 3.0
    occupied: set[tuple[int, int]] = set()
    for x, y in pts:
        cx = min(2, max(0, int((x - x0) / w))) if w > 0 else 0
        cy = min(2, max(0, int((y - y0) / h))) if h > 0 else 0
        occupied.add((cx, cy))
    return len(occupied) / 9.0


def frac_within(pts: list[tuple[float, float]], center: tuple[float, float],
                radius: float) -> float:
    """Fraction of members within *radius* of *center*."""
    n = len(pts)
    if n == 0:
        return 0.0
    r2 = radius * radius
    c = 0
    for x, y in pts:
        dx = x - center[0]
        dy = y - center[1]
        if dx * dx + dy * dy <= r2:
            c += 1
    return c / n


def measure(sim: CrowdSimulator) -> dict[str, float]:
    pts = _positions(sim)
    return {
        "r50_m": r50_m(pts),
        "densest_cell_frac": densest_cell_frac(pts),
        "peak_local_density": peak_local_density(pts),
        "sector_occupancy_frac": sector_occupancy_frac(pts, sim.bounds),
        "n": float(len(pts)),
    }


# ---------------------------------------------------------------------------
# Scenarios.
# ---------------------------------------------------------------------------

BOUNDS = (0.0, 0.0, 200.0, 200.0)
STEADY_TICKS = 200
DT = 0.1


def run_seeded_riot() -> dict[str, float]:
    random.seed(42)
    sim = _build_riot(bounds=BOUNDS, seed=42)
    for _ in range(STEADY_TICKS):
        sim.tick(DT)
    return measure(sim)


def run_operator_spawn() -> dict[str, float]:
    """Faithful reproduction of WorldBuilder.spawn_crowd (_world.py ~line 309):
    one CrowdSimulator, ONE spawn_crowd at centre, NO anchor, NO seed/rng."""
    random.seed(42)
    sim = CrowdSimulator(bounds=BOUNDS, max_members=500)
    sim.spawn_crowd(center=(100.0, 100.0), count=120, radius=16.0,
                    mood=CrowdMood.RIOTING)
    for _ in range(STEADY_TICKS):
        sim.tick(DT)
    return measure(sim)


def run_central_surge() -> dict[str, float]:
    """From seeded_riot steady state, inject ONE high-intensity event at the
    centre. Record pile fraction at peak, then let it decay and measure again."""
    random.seed(42)
    sim = _build_riot(bounds=BOUNDS, seed=42)
    for _ in range(STEADY_TICKS):
        sim.tick(DT)

    center = (100.0, 100.0)
    # A throw_object event is what the riot preset itself uses as a flashpoint;
    # high intensity + radius ~30 so it strongly pulls RIOTING members in
    # (_move_members surges to events within 25 m).
    sim.inject_event(CrowdEvent(
        event_type="throw_object",
        position=center,
        radius=30.0,
        intensity=1.0,
        timestamp=sim._time,
    ))
    # Peak: tick 100 while the event is fresh / re-injected each segment? No —
    # single injection, let it act. Measure the worst (peak) pile over the
    # window so we capture the maximum convergence, not just the endpoint.
    peak = 0.0
    for _ in range(100):
        sim.tick(DT)
        f = frac_within(_positions(sim), center, 10.0)
        if f > peak:
            peak = f

    # Decay: event intensity falls (_EVENT_DECAY_RATE) and is removed when
    # intensity <= 0.01; tick 100 more and measure the residual pile.
    for _ in range(100):
        sim.tick(DT)
    after = frac_within(_positions(sim), center, 10.0)

    m = measure(sim)
    m["event_pile_frac_peak"] = peak
    m["event_pile_frac_after"] = after
    return m


# ---------------------------------------------------------------------------
# Pytest entry points (assertions are loose sanity bounds, not the verdict —
# the verdict comes from printed numbers).
# ---------------------------------------------------------------------------

def test_seeded_riot_spreads():
    m = run_seeded_riot()
    # Good baseline should NOT be a single pile.
    assert m["densest_cell_frac"] < 0.5
    assert m["sector_occupancy_frac"] >= 0.5


def test_operator_spawn_measured():
    m = run_operator_spawn()
    # No assertion on the verdict — just that it produced members.
    assert m["n"] > 0


def test_operator_spawn_no_longer_piles():
    """RED/GREEN guard for the cohesion-to-distributed-anchors fix
    (riot legibility rework B, 2026-06-15).

    Phase-1 measured the operator path as a central pile: r50 ~9 m, densest
    10 m cell 17-20%, peak local density ~20, only 1/9 sectors occupied. With
    the auto-fan into distinct sector sub-clusters the same anchorless RIOTING
    spawn must spread across the map. These bounds are well inside the measured
    after-fix numbers (r50 ~55, densest ~0.05, peak ~6, sector ~0.78) and well
    outside the pre-fix pile, so this is RED without the fix and GREEN with it.
    """
    m = run_operator_spawn()
    # Spread out, not a tight central knot.
    assert m["r50_m"] >= 30.0, f"r50 too tight (pile): {m['r50_m']}"
    # No single 10 m cell hoards the crowd.
    assert m["densest_cell_frac"] <= 0.10, f"densest cell too hot: {m['densest_cell_frac']}"
    # Local density capped like the good baseline (was ~20).
    assert m["peak_local_density"] <= 12.0, f"peak density too high: {m['peak_local_density']}"
    # Uses most of the city (was 1/9).
    assert m["sector_occupancy_frac"] >= 0.55, f"too few sectors used: {m['sector_occupancy_frac']}"


def test_central_surge_measured():
    m = run_central_surge()
    assert m["event_pile_frac_peak"] >= 0.0
    assert m["event_pile_frac_after"] >= 0.0


def test_spawn_crowd_fans_into_distinct_groups():
    """A large anchorless spawn must fan into multiple DISTINCT group_ids, each
    with its OWN sector anchor (cohesion-to-distributed-anchors). RED on the old
    single-group path (which produced exactly ONE group_id / no anchors)."""
    random.seed(7)
    sim = CrowdSimulator(bounds=BOUNDS, max_members=500)
    ids = sim.spawn_crowd(center=(100.0, 100.0), count=120, radius=16.0,
                          mood=CrowdMood.RIOTING)
    assert len(ids) == 120, "every requested member must be spawned"
    group_ids = {m.group_id for m in sim.members}
    assert len(group_ids) >= 5, f"expected ~6 sector groups, got {len(group_ids)}"
    # Every member carries a stable sector anchor (none falls back to centroid).
    assert all(m.anchor is not None for m in sim.members)
    # The sector anchors are genuinely distinct points across the map.
    anchors = {m.anchor for m in sim.members}
    assert len(anchors) >= 5, f"expected distinct sector anchors, got {len(anchors)}"


def test_small_calm_crowd_unchanged_single_group():
    """Below the auto-fan threshold (and with no explicit sectors) a small crowd
    keeps the original single-cluster behaviour — no anchors, one group_id.
    Guards against over-fanning small/calm gatherings."""
    random.seed(7)
    sim = CrowdSimulator(bounds=BOUNDS, max_members=500)
    sim.spawn_crowd(center=(100.0, 100.0), count=20, radius=8.0,
                    mood=CrowdMood.CALM)
    group_ids = {m.group_id for m in sim.members}
    assert len(group_ids) == 1, "small calm crowd must stay one cluster"
    assert all(m.anchor is None for m in sim.members)


def test_large_calm_crowd_does_not_fan():
    """Round-1 fix (2026-06-15): auto-fan is VOLATILE-ONLY. A CALM crowd just
    OVER the count threshold — and a big 500 CALM crowd — must NOT fragment into
    6 knots. This is the over-fanning gap the count=20 guard missed, and the
    cause of the perf regression (tight knots super-linearly raise separation
    cost). RED before the mood gate (CALM>40 fanned into 6 groups, anchored),
    GREEN after (one group, no anchors)."""
    for count in (41, 500):
        random.seed(7)
        sim = CrowdSimulator(bounds=BOUNDS, max_members=600)
        sim.spawn_crowd(center=(100.0, 100.0), count=count, radius=45.0,
                        mood=CrowdMood.CALM)
        group_ids = {m.group_id for m in sim.members}
        assert len(group_ids) == 1, (
            f"calm {count}-member crowd must stay one cluster, got {len(group_ids)}"
        )
        assert all(m.anchor is None for m in sim.members), (
            f"calm {count}-member crowd must not be anchored into sectors"
        )


def test_panicked_crowd_does_not_fan():
    """A PANICKED crowd (stampede) is not a volatile auto-fan mood, so even a
    large one stays a single radial blob that explodes from the flashpoint —
    not 6 scattered sector knots. RED before the mood gate (300>40 fanned)."""
    random.seed(7)
    sim = CrowdSimulator(bounds=BOUNDS, max_members=500)
    sim.spawn_crowd(center=(100.0, 100.0), count=300, radius=35.0,
                    mood=CrowdMood.PANICKED)
    group_ids = {m.group_id for m in sim.members}
    assert len(group_ids) == 1, f"stampede must stay one blob, got {len(group_ids)}"
    assert all(m.anchor is None for m in sim.members)


def test_build_stampede_single_blob():
    """_build_stampede preset must remain ONE 300-member blob (not 6 knots).
    Guards the preset against silent scope-creep from the auto-fan."""
    sim = _build_stampede(BOUNDS)
    group_ids = {m.group_id for m in sim.members}
    assert len(group_ids) == 1, f"stampede preset fanned: {len(group_ids)} groups"
    assert all(m.anchor is None for m in sim.members)
    assert len(sim.members) == 300


def test_build_standoff_single_line():
    """_build_standoff preset is AGITATED (a volatile mood) but represents ONE
    crowd massed in front of a single police line. It opts out via sectors=1, so
    it must stay one cluster, not fan into 6 scattered knots. RED if the standoff
    preset loses its sectors=1 opt-out."""
    sim = _build_standoff(BOUNDS)
    group_ids = {m.group_id for m in sim.members}
    assert len(group_ids) == 1, f"standoff preset fanned: {len(group_ids)} groups"
    assert all(m.anchor is None for m in sim.members)
    assert len(sim.members) == 100


def test_world_operator_riot_spreads():
    """End-to-end through the real operator path: World.spawn_crowd (_world.py
    ~L309) with a RIOTING crowd must spread across the map (sector_occupancy
    high, no central pile). This is the exact path the operator UI exercises.
    RED on the old wiring (one anchorless group -> central pile)."""
    from tritium_lib.sim_engine.world import World, WorldConfig

    random.seed(7)
    world = World(WorldConfig(map_size=(200.0, 200.0)))
    world.spawn_crowd((100.0, 100.0), count=120, radius=16.0, mood=CrowdMood.RIOTING)
    for _ in range(STEADY_TICKS):
        world.crowd.tick(DT)
    m = measure(world.crowd)
    assert m["r50_m"] >= 30.0, f"operator riot piled: r50={m['r50_m']}"
    assert m["densest_cell_frac"] <= 0.10, f"operator riot hot cell: {m['densest_cell_frac']}"
    assert m["sector_occupancy_frac"] >= 0.55, f"operator riot one corner: {m['sector_occupancy_frac']}"


# ---------------------------------------------------------------------------
# Standalone report.
# ---------------------------------------------------------------------------

def _print(name: str, m: dict[str, float]) -> None:
    print(f"\n=== {name} (n={int(m['n'])}) ===")
    print(f"  r50_m                 = {m['r50_m']:.2f}")
    print(f"  densest_cell_frac     = {m['densest_cell_frac']:.4f}")
    print(f"  peak_local_density    = {m['peak_local_density']:.1f}")
    print(f"  sector_occupancy_frac = {m['sector_occupancy_frac']:.4f}")
    if "event_pile_frac_peak" in m:
        print(f"  event_pile_frac_peak  = {m['event_pile_frac_peak']:.4f}")
        print(f"  event_pile_frac_after = {m['event_pile_frac_after']:.4f}")


if __name__ == "__main__":
    _print("seeded_riot (GOOD baseline)", run_seeded_riot())
    _print("operator_spawn (SUSPECT)", run_operator_spawn())
    _print("central_surge", run_central_surge())

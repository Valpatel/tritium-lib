# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Organic crowd movement (riot-quality rework, 2026-06-14).

The live riot rendered as "lines of units in odd positions": fleeing crowds
streamed to 4 boundary-midpoint exits, agitated/rioting crowds snapped onto a
single group center, weak separation let members stack, and raw per-tick
velocity made motion jittery. These tests pin the fixes that make the crowd
mill and disperse organically.
"""

from __future__ import annotations

import random

from tritium_lib.sim_engine.crowd import (
    CrowdSimulator, CrowdMember, CrowdMood, CrowdEvent,
    _SPEED_FLEEING, _build_riot,
)
from tritium_lib.sim_engine.ai.steering import magnitude

BOUNDS = (-100.0, -100.0, 100.0, 100.0)


def test_exits_are_distributed_not_four():
    """Fleeing must disperse to many exit zones, not collapse onto 4 lines."""
    sim = CrowdSimulator(BOUNDS)
    assert len(sim._compute_exits()) > 4


def test_separation_repels_a_mid_distance_neighbor():
    """Stronger separation (>1.5 m) keeps members from stacking into lines."""
    sim = CrowdSimulator(BOUNDS)
    a = CrowdMember(member_id="a", position=(0.0, 0.0))
    b = CrowdMember(member_id="b", position=(2.5, 0.0))  # inside new radius, outside the old 1.5 m
    sim.members = [a, b]
    sim._grid.rebuild(sim.members)
    f = sim._separation_force(a)
    assert magnitude(f) > 0.0, "a 2.5 m neighbor should now produce repulsion"
    assert f[0] < 0.0, "a should be pushed away from b"


def test_velocity_is_smoothed_from_rest():
    """Velocity ramps (exponential smoothing) instead of snapping to full speed."""
    sim = CrowdSimulator(BOUNDS)
    m = CrowdMember(member_id="m", position=(0.0, 0.0), mood=CrowdMood.FLEEING, fear=0.9)
    sim.members = [m]
    sim._grid.rebuild(sim.members)
    sim._group_center_cache = {}
    sim._move_members(0.1)
    assert 0.0 < magnitude(m.velocity) < _SPEED_FLEEING * 0.6, (
        "first-tick speed should be a fraction of full (smoothed), not a snap"
    )


def test_fleeing_disperses_across_many_exit_zones():
    """A panicked crowd heads to many distinct exit zones, not 4 streams."""
    sim = CrowdSimulator(BOUNDS, max_members=500)
    sim.spawn_crowd((0.0, 0.0), 120, radius=30.0, mood=CrowdMood.PANICKED, leader_ratio=0.0)
    sim.inject_event(CrowdEvent("gunshot", (0.0, 0.0), 60.0, 1.0, 0.0))
    for _ in range(6):
        sim.tick(0.5)
    targets = {m.exit_target for m in sim.members if getattr(m, "exit_target", None)}
    assert len(targets) > 4, f"fleeing collapsed onto {len(targets)} exits (want many)"


def test_riot_does_not_collapse_to_a_point():
    """Weak sanity check only. NOTE: bbox spread != legibility — a crowd can have
    a wide bounding box yet still be an unreadable pile in the centre (two outlier
    members stretch the bbox while everyone else stacks). The load-bearing metric
    is OCCUPANCY (see test_riot_occupancy_is_spread_not_a_central_pile)."""
    sim = _build_riot(BOUNDS)
    for _ in range(30):
        sim.tick(0.5)
    xs = [m.position[0] for m in sim.members]
    ys = [m.position[1] for m in sim.members]
    spread_x, spread_y = max(xs) - min(xs), max(ys) - min(ys)
    assert spread_x > 40.0 and spread_y > 40.0, (
        f"riot collapsed to a blob: {spread_x:.1f} x {spread_y:.1f} m"
    )


def _occupancy(sim, cell_size=20.0):
    """Bin members into *cell_size* m cells. Returns (busiest_count, n_members,
    occupied_cell_count)."""
    cells: dict[tuple[int, int], int] = {}
    for m in sim.members:
        key = (int(m.position[0] // cell_size), int(m.position[1] // cell_size))
        cells[key] = cells.get(key, 0) + 1
    busiest = max(cells.values()) if cells else 0
    return busiest, len(sim.members), len(cells)


def test_riot_occupancy_is_spread_not_a_central_pile():
    """LOAD-BEARING legibility metric (riot rework, 2026-06-15): bin the riot
    into 20 m cells and require that NO single cell holds a large share of the
    crowd AND that members occupy MANY distinct cells — at spawn AND after the
    crowd has marched/held/rotated. This is what the operator complaint ("a
    massive pile of units only in the centre") actually demands; bbox spread
    (test_riot_does_not_collapse_to_a_point) does not capture it.

    Thresholds are set from MEASURED post-fix values with wide headroom:
      busiest cell  9.5% / 8.0% / 7.5% at t=0/30/90s  -> assert < 20%
      occupied      24   / 26   / 24   cells          -> assert >= 12
    Determinism: _build_riot(seed=...) seeds placement + objective RNG, and we
    seed the module RNG (movement jitter) — so these numbers are repeatable.
    """
    random.seed(20260615)
    sim = _build_riot(BOUNDS, seed=1234)
    cell = 20.0
    max_share = 0.20
    min_cells = 12

    def check(label):
        busiest, n, occupied = _occupancy(sim, cell)
        share = busiest / n
        assert share < max_share, (
            f"{label}: busiest cell holds {share:.1%} of the crowd "
            f"({busiest}/{n}) — a central pile (want < {max_share:.0%})"
        )
        assert occupied >= min_cells, (
            f"{label}: only {occupied} cells occupied (want >= {min_cells}) — "
            f"crowd is clumped, not spread across the map"
        )

    check("t=0")
    for _ in range(60):     # 30 sim-seconds
        sim.tick(0.5)
    check("t=30s")
    for _ in range(120):    # to 90 sim-seconds
        sim.tick(0.5)
    check("t=90s")


def test_riot_preset_spawns_scattered_subgroups():
    """The riot preset seeds several distinct groups (organic clusters), not one blob."""
    sim = _build_riot(BOUNDS)
    group_ids = {m.group_id for m in sim.members if m.group_id}
    assert len(group_ids) >= 3, f"expected scattered subgroups, got {len(group_ids)} group(s)"


def _square_obstacles(half=20.0):
    """A single square building footprint centred at the origin (real API)."""
    from tritium_lib.tracking.obstacles import BuildingObstacles
    obs = BuildingObstacles()
    obs.polygons = [[(-half, -half), (half, -half), (half, half), (-half, half)]]
    obs._heights = [12.0]
    obs._compute_aabbs()
    return obs


def test_members_never_walk_into_buildings():
    """With building obstacles, rioters drawn toward an event INSIDE a building
    must stop at the wall — never inside the footprint (the through-buildings bug)."""
    obs = _square_obstacles(20.0)
    sim = CrowdSimulator(BOUNDS, max_members=500, obstacles=obs)
    # Spawn east of the building, heading west toward an event at its centre.
    sim.spawn_crowd((40.0, 0.0), 40, radius=8.0, mood=CrowdMood.RIOTING, leader_ratio=0.05)
    sim.inject_event(CrowdEvent("throw_object", (0.0, 0.0), 10.0, 0.6, 0.0))
    inside_ticks = 0
    for _ in range(40):
        sim.tick(0.5)
        for m in sim.members:
            if obs.point_in_building(m.position[0], m.position[1]):
                inside_ticks += 1
    assert inside_ticks == 0, f"{inside_ticks} member-ticks inside a building (must be 0)"


def test_member_slides_along_wall_instead_of_piling():
    """A member pushed diagonally into a wall slides along the free axis (flows
    around the building) rather than hard-stopping and piling at the wall."""
    from tritium_lib.tracking.obstacles import BuildingObstacles
    obs = BuildingObstacles()
    obs.polygons = [[(0.0, -40.0), (40.0, -40.0), (40.0, 40.0), (0.0, 40.0)]]  # wall at x=0, building x>0
    obs._heights = [12.0]
    obs._compute_aabbs()
    sim = CrowdSimulator(BOUNDS, max_members=10, obstacles=obs)
    m = CrowdMember(member_id="m", position=(-1.0, 0.0), velocity=(5.0, 5.0))
    sim._apply_velocity(m, 0.5)   # diagonal toward +x (into building) and +y (free)
    assert not obs.point_in_building(m.position[0], m.position[1]), "must not be inside the building"
    assert m.position[0] <= 0.0 + 1e-6, "blocked x move was rejected (stayed out of the building)"
    assert m.position[1] > 0.5, "slid along the free +y axis instead of stopping dead"


def test_exits_prefer_perimeter_street_nodes():
    """With a road graph, flee exits are real PERIMETER street nodes (roads out of
    the area), so a fleeing crowd disperses DOWN actual streets — not arbitrary
    geometric boundary points. Makes 'scatters down the streets' literally true."""
    # BOUNDS radius is 100; perimeter band starts at 55.
    central = [(0.0, 0.0), (5.0, 5.0), (-8.0, 3.0)]
    perimeter = [(90.0, 0.0), (-90.0, 0.0), (0.0, 90.0), (0.0, -90.0), (70.0, 70.0)]
    sim = CrowdSimulator(BOUNDS, street_nodes=central + perimeter)
    exits = sim._exits
    assert len(exits) >= 4
    for ex in exits:
        assert magnitude(ex) >= 55.0, f"exit {ex} is not on the perimeter"
    for c in central:
        assert c not in exits, f"central node {c} must not be an exit"


def test_set_street_nodes_recomputes_exits():
    """Live path: street data arrives after construction. set_street_nodes must
    swap the geometric ring exits for perimeter street exits."""
    sim = CrowdSimulator(BOUNDS)  # no streets -> geometric ring
    ring = list(sim._exits)
    sim.set_street_nodes([(90.0, 0.0), (-90.0, 0.0), (0.0, 90.0), (0.0, -90.0), (80.0, 80.0)])
    assert sim._exits != ring, "exits should be recomputed from street nodes"
    for ex in sim._exits:
        assert magnitude(ex) >= 55.0


def test_member_inside_building_is_ejected():
    """A member that STARTS a tick inside a building (spawned/snapped in, or a
    gather point that sat on a building node) must be ejected to open space, not
    frozen inside forever. Wall-slide only stops members ENTERING — it can't free
    one already inside (all slide axes are blocked → velocity zeroed → stuck)."""
    obs = _square_obstacles(20.0)  # solid building covering [-20,20] x [-20,20]
    sim = CrowdSimulator(BOUNDS, max_members=10, obstacles=obs)
    m = CrowdMember(member_id="stuck", position=(0.0, 0.0), velocity=(3.0, 0.0))
    assert obs.point_in_building(*m.position), "precondition: member is inside the building"
    sim._apply_velocity(m, 0.5)
    assert not obs.point_in_building(*m.position), (
        f"member at {m.position} is still inside the building (not ejected)"
    )


def test_group_center_snaps_to_a_street_node():
    """Group gather points snap to the nearest real street node, so crowds form
    on streets/junctions instead of collapsing onto an arbitrary centroid."""
    nodes = [(30.0, 0.0), (-30.0, 0.0), (0.0, 40.0)]
    sim = CrowdSimulator(BOUNDS, max_members=200, street_nodes=nodes)
    sim.spawn_crowd((10.0, 0.0), 20, radius=4.0, mood=CrowdMood.AGITATED, leader_ratio=0.0)
    sim.tick(0.5)  # computes (and snaps) the group centers
    centers = list(sim._group_center_cache.values())
    assert centers, "a group center was computed"
    assert all(c in nodes for c in centers), f"group center not on a street node: {centers}"
    assert (30.0, 0.0) in centers, "should snap to the nearest node to the (~10,0) centroid"


def test_no_obstacles_members_still_move():
    """obstacles=None preserves the original flat-plane behaviour (no regression)."""
    sim = CrowdSimulator(BOUNDS, max_members=500)  # obstacles defaults None
    sim.spawn_crowd((0.0, 0.0), 20, radius=10.0, mood=CrowdMood.RIOTING, leader_ratio=0.05)
    sim.inject_event(CrowdEvent("throw_object", (5.0, 5.0), 10.0, 0.6, 0.0))
    before = [m.position for m in sim.members]
    for _ in range(5):
        sim.tick(0.5)
    moved = sum(1 for m, b in zip(sim.members, before) if m.position != b)
    assert moved > 0, "members should still move when there are no obstacles"

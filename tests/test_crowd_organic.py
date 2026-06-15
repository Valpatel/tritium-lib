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
    """The scattered riot preset stays spread across the area (organic clusters
    clashing locally), not a single dense blob rushing one central point."""
    sim = _build_riot(BOUNDS)
    for _ in range(30):
        sim.tick(0.5)
    xs = [m.position[0] for m in sim.members]
    ys = [m.position[1] for m in sim.members]
    spread_x, spread_y = max(xs) - min(xs), max(ys) - min(ys)
    assert spread_x > 40.0 and spread_y > 40.0, (
        f"riot collapsed to a blob: {spread_x:.1f} x {spread_y:.1f} m"
    )


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

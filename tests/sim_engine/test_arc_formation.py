# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Pure-geometry tests for the ARC (kettle cordon) formation.

ARC places members evenly along a circular arc around ``leader_pos`` (the
cordon CENTRE) that spans ``360deg - gap_angle`` centred OPPOSITE ``facing``,
so a wedge of width ``gap_angle`` around ``facing`` (the dispersal gap) is left
empty.  These tests pin slot count, radius, and the empty gap direction; the
helper is deterministic and free of any game-engine state.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.sim_engine.ai.formations import (
    FormationConfig,
    FormationType,
    get_formation_positions,
)


def _angles_about(center, slots):
    """World bearing (radians) of each slot as seen from the arc centre."""
    return [math.atan2(s[1] - center[1], s[0] - center[0]) for s in slots]


def _ang_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference between two bearings (radians)."""
    d = (a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def test_arc_slot_count_matches_members():
    for n in (1, 2, 5, 8, 20):
        config = FormationConfig(
            formation_type=FormationType.ARC,
            leader_pos=(10.0, -4.0),
            num_members=n,
            radius=9.0,
        )
        slots = get_formation_positions(config)
        assert len(slots) == n


def test_arc_explicit_radius_places_all_slots_on_ring():
    center = (5.0, 5.0)
    radius = 12.0
    config = FormationConfig(
        formation_type=FormationType.ARC,
        leader_pos=center,
        facing=0.7,
        num_members=9,
        radius=radius,
    )
    slots = get_formation_positions(config)
    for s in slots:
        d = math.hypot(s[0] - center[0], s[1] - center[1])
        assert d == pytest.approx(radius, abs=1e-6)


def test_arc_default_radius_from_spacing_clamped_to_min():
    # spacing * n / (2*pi) below 6.0 -> clamps to the 6.0 m floor.
    small = FormationConfig(
        formation_type=FormationType.ARC,
        leader_pos=(0.0, 0.0),
        spacing=2.0,
        num_members=4,  # 2*4/(2pi) ~= 1.27 -> clamped
    )
    slots = get_formation_positions(small)
    for s in slots:
        assert math.hypot(s[0], s[1]) == pytest.approx(6.0, abs=1e-6)

    # A large ring exceeds the floor and uses the derived radius.
    big = FormationConfig(
        formation_type=FormationType.ARC,
        leader_pos=(0.0, 0.0),
        spacing=5.0,
        num_members=30,  # 5*30/(2pi) ~= 23.87
    )
    expected = 5.0 * 30 / (2 * math.pi)
    for s in get_formation_positions(big):
        assert math.hypot(s[0], s[1]) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("facing", [0.0, 1.2, math.pi, -2.0, 2.9])
def test_arc_gap_is_empty_of_slots(facing):
    """No slot falls inside the +/- gap/2 wedge centred on the facing bearing."""
    gap_angle = 75.0
    half_gap = math.radians(gap_angle) / 2.0
    center = (3.0, -2.0)
    config = FormationConfig(
        formation_type=FormationType.ARC,
        leader_pos=center,
        facing=facing,
        num_members=8,
        radius=10.0,
        gap_angle=gap_angle,
    )
    slots = get_formation_positions(config)
    for ang in _angles_about(center, slots):
        # Every slot is at least half the gap away from the gap direction.
        assert _ang_diff(ang, facing) >= half_gap - 1e-6


def test_arc_reaches_both_gap_edges_and_is_symmetric():
    """The arc spans right up to the gap edges and is symmetric about the gap."""
    facing = 0.0
    gap_angle = 90.0
    half_gap = math.radians(gap_angle) / 2.0
    center = (0.0, 0.0)
    config = FormationConfig(
        formation_type=FormationType.ARC,
        leader_pos=center,
        facing=facing,
        num_members=10,
        radius=8.0,
        gap_angle=gap_angle,
    )
    angles = _angles_about(center, get_formation_positions(config))
    # The two end slots sit exactly on the gap edges (gap/2 off facing).
    nearest = min(_ang_diff(a, facing) for a in angles)
    assert nearest == pytest.approx(half_gap, abs=1e-6)
    # Symmetric about facing: for every slot there is a mirror-image slot.
    signed = sorted(((a - facing + math.pi) % (2 * math.pi) - math.pi) for a in angles)
    for lo, hi in zip(signed, reversed(signed)):
        assert lo == pytest.approx(-hi, abs=1e-6)


def test_arc_single_member_sits_opposite_the_gap():
    center = (4.0, 4.0)
    facing = 0.0  # gap opens toward +X
    config = FormationConfig(
        formation_type=FormationType.ARC,
        leader_pos=center,
        facing=facing,
        num_members=1,
        radius=7.0,
    )
    (slot,) = get_formation_positions(config)
    # Directly opposite the gap: due west of the centre at the ring radius.
    assert slot[0] == pytest.approx(center[0] - 7.0, abs=1e-6)
    assert slot[1] == pytest.approx(center[1], abs=1e-6)

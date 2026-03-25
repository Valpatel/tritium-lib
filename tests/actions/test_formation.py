# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.actions.formation."""

import math
from tritium_lib.actions.formation import (
    VALID_FORMATIONS,
    VALID_ORDERS,
    RALLY_RADIUS,
    compute_formation_offsets,
    compute_scatter_positions,
    is_within_rally_radius,
)


def test_valid_formations():
    """VALID_FORMATIONS contains expected types."""
    assert "wedge" in VALID_FORMATIONS
    assert "line" in VALID_FORMATIONS
    assert "column" in VALID_FORMATIONS
    assert "circle" in VALID_FORMATIONS


def test_valid_orders():
    """VALID_ORDERS contains expected orders."""
    assert "advance" in VALID_ORDERS
    assert "hold" in VALID_ORDERS
    assert "retreat" in VALID_ORDERS


def test_line_formation():
    """Line formation produces horizontal offsets."""
    offsets = compute_formation_offsets("line", 3)
    assert len(offsets) == 3
    # All y offsets should be 0
    for _, y in offsets:
        assert y == 0.0


def test_column_formation():
    """Column formation produces vertical offsets."""
    offsets = compute_formation_offsets("column", 3)
    assert len(offsets) == 3
    # All x offsets should be 0
    for x, _ in offsets:
        assert x == 0.0
    # Leader at front (y=0), others behind
    assert offsets[0][1] == 0.0
    assert offsets[1][1] < 0.0


def test_wedge_formation():
    """Wedge formation has leader at point."""
    offsets = compute_formation_offsets("wedge", 5)
    assert len(offsets) == 5
    assert offsets[0] == (0.0, 0.0)  # Leader at center


def test_circle_formation():
    """Circle formation distributes units around center."""
    offsets = compute_formation_offsets("circle", 4)
    assert len(offsets) == 4
    # All should be equidistant from origin
    dists = [math.hypot(x, y) for x, y in offsets]
    assert max(dists) - min(dists) < 0.01


def test_single_unit():
    """Single unit gets (0, 0) offset."""
    offsets = compute_formation_offsets("line", 1)
    assert offsets == [(0.0, 0.0)]


def test_zero_units():
    """Zero units returns empty list."""
    assert compute_formation_offsets("line", 0) == []


def test_scatter_positions():
    """compute_scatter_positions returns correct count."""
    positions = compute_scatter_positions((10.0, 20.0), 5)
    assert len(positions) == 5
    # All should be away from center
    for x, y in positions:
        dist = math.hypot(x - 10.0, y - 20.0)
        assert dist >= 7.0  # At least min_dist - some tolerance


def test_is_within_rally_radius():
    """is_within_rally_radius correctly checks distance."""
    assert is_within_rally_radius((0, 0), (0, 0))
    assert is_within_rally_radius((10, 0), (0, 0))
    assert not is_within_rally_radius((50, 0), (0, 0))
    assert is_within_rally_radius((20, 20), (0, 0), radius=50)

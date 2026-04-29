# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Wave 200 security tests — RF motion target poisoning resistance.

Proves that ``TargetTracker.update_from_rf_motion`` rejects:
  - position values containing NaN or +/-Inf (HIGH — slips past
    the (0, 0) reject because NaN compares False to everything),
  - non-numeric / unparsable position dicts and tuples,
  - empty target_id (regression).

Also proves the prior (0, 0) behavior still rejects.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.tracking.target_tracker import TargetTracker


def _new_tracker() -> TargetTracker:
    return TargetTracker()


def test_rejects_position_zero_zero() -> None:
    """Pre-existing behavior — (0, 0) is still rejected."""
    t = _new_tracker()
    t.update_from_rf_motion({
        "target_id": "rfm_a_b",
        "position": (0.0, 0.0),
        "confidence": 0.9,
        "direction_hint": "approaching",
        "pair_id": "a::b",
    })
    assert t.get_target("rfm_a_b") is None


def test_rejects_nan_position_tuple() -> None:
    """A NaN-poisoned position must be dropped — slips past (0, 0) check."""
    t = _new_tracker()
    nan = float("nan")
    t.update_from_rf_motion({
        "target_id": "rfm_a_b",
        "position": (nan, nan),
        "confidence": 0.9,
        "direction_hint": "approaching",
        "pair_id": "a::b",
    })
    assert t.get_target("rfm_a_b") is None


def test_rejects_nan_position_dict() -> None:
    """A NaN in a dict-shaped position must be dropped."""
    t = _new_tracker()
    nan = float("nan")
    t.update_from_rf_motion({
        "target_id": "rfm_a_b",
        "position": {"x": nan, "y": 1.0},
        "confidence": 0.9,
        "direction_hint": "approaching",
        "pair_id": "a::b",
    })
    assert t.get_target("rfm_a_b") is None


def test_rejects_inf_position() -> None:
    """+/-Inf positions must be dropped (would propagate through arithmetic)."""
    t = _new_tracker()
    inf = float("inf")
    t.update_from_rf_motion({
        "target_id": "rfm_inf_pos",
        "position": (inf, 1.0),
        "confidence": 0.9,
        "direction_hint": "approaching",
        "pair_id": "a::b",
    })
    assert t.get_target("rfm_inf_pos") is None

    t.update_from_rf_motion({
        "target_id": "rfm_neg_inf",
        "position": (1.0, -inf),
        "confidence": 0.9,
        "direction_hint": "approaching",
        "pair_id": "a::b",
    })
    assert t.get_target("rfm_neg_inf") is None


def test_rejects_unparsable_string_position() -> None:
    """Non-numeric strings in dict-shaped position must be dropped silently."""
    t = _new_tracker()
    t.update_from_rf_motion({
        "target_id": "rfm_str",
        "position": {"x": "not a number", "y": 0.5},
        "confidence": 0.9,
        "direction_hint": "approaching",
        "pair_id": "a::b",
    })
    assert t.get_target("rfm_str") is None


def test_rejects_empty_target_id() -> None:
    """Regression — empty target_id is still dropped."""
    t = _new_tracker()
    t.update_from_rf_motion({
        "target_id": "",
        "position": (10.0, 20.0),
        "confidence": 0.9,
        "direction_hint": "approaching",
        "pair_id": "a::b",
    })
    # Nothing to assert by name, but get_all should be empty.
    assert len(t.get_all()) == 0


def test_accepts_finite_nonzero_position() -> None:
    """Sanity — finite non-zero positions still create targets."""
    t = _new_tracker()
    t.update_from_rf_motion({
        "target_id": "rfm_ok",
        "position": (10.5, 20.25),
        "confidence": 0.9,
        "direction_hint": "approaching",
        "pair_id": "a::b",
    })
    target = t.get_target("rfm_ok")
    assert target is not None
    assert target.position == (10.5, 20.25)
    assert math.isfinite(target.position[0])
    assert math.isfinite(target.position[1])

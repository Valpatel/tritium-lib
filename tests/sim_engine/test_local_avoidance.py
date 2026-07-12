# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Reusable inter-agent local avoidance (separation) — the REACTIVE movement
layer. Units must push apart when crowding so a dense crowd jostles instead of
ghosting through itself, while normally-spaced units are untouched, and every
nudge is bounded (no teleport)."""
import math

from tritium_lib.sim_engine.ai.local_avoidance import separation_deltas


def _grid(points):
    """Trivial neighbor query over a fixed point set (id->pos)."""
    def q(pos, radius):
        out = []
        for p in points:
            d = math.hypot(p[0] - pos[0], p[1] - pos[1])
            if d <= radius:
                out.append(p)
        return out
    return q


def test_no_push_when_well_spaced():
    agents = [("a", (0.0, 0.0)), ("b", (10.0, 0.0))]
    q = _grid([(0.0, 0.0), (10.0, 0.0)])
    deltas = separation_deltas(agents, q, desired_separation=1.6, max_push=0.35)
    assert deltas == {}, f"well-spaced agents should not be nudged, got {deltas}"


def test_overlapping_pair_pushed_apart():
    # Two agents almost on top of each other -> pushed in OPPOSITE directions.
    agents = [("a", (0.0, 0.0)), ("b", (0.4, 0.0))]
    q = _grid([(0.0, 0.0), (0.4, 0.0)])
    deltas = separation_deltas(agents, q, desired_separation=1.6, max_push=0.35)
    assert "a" in deltas and "b" in deltas
    # a is left of b -> a pushed -x, b pushed +x
    assert deltas["a"][0] < 0 and deltas["b"][0] > 0, deltas


def test_push_is_bounded():
    agents = [("a", (0.0, 0.0)), ("b", (0.01, 0.0))]  # near-coincident -> huge raw force
    q = _grid([(0.0, 0.0), (0.01, 0.0)])
    deltas = separation_deltas(agents, q, desired_separation=2.0, max_push=0.3)
    for d in deltas.values():
        assert math.hypot(*d) <= 0.3 + 1e-9, f"push exceeded max_push: {d}"


def test_self_position_in_neighbors_is_ignored():
    # neighbors_fn returns the agent's OWN position too — must not self-repel.
    agents = [("a", (5.0, 5.0))]
    q = _grid([(5.0, 5.0)])  # only itself nearby
    deltas = separation_deltas(agents, q, desired_separation=1.6, max_push=0.35)
    assert deltas == {}, f"a lone agent must not move, got {deltas}"


def test_dense_crowd_all_get_bounded_nudges():
    pts = [(float(i) * 0.5, 0.0) for i in range(6)]  # 6 agents 0.5m apart in a line
    agents = [(f"u{i}", p) for i, p in enumerate(pts)]
    q = _grid(pts)
    deltas = separation_deltas(agents, q, desired_separation=1.6, max_push=0.4)
    assert len(deltas) >= 4, f"interior crowded agents should be nudged, got {len(deltas)}"
    for d in deltas.values():
        assert math.hypot(*d) <= 0.4 + 1e-9

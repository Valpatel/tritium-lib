# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.geo.collider_shape — colliders a hull solver can chew.

The failure this pins down cost the Newton lane four ticks, and it looks like
nothing.  A ground plane authored the obvious way — a 100 x 100 quad, four
corners, all at z = 0 — is handed to a physics backend that convex-hulls every
collision shape.  qhull is asked to build an initial simplex from a point set
of rank 2 and answers "Initial simplex is flat (facet 1 is coplanar with the
interior point)".  The exception unwinds through the physics extension's
initializer, which had already set an `_initializing` guard and never clears it
on the error path.  From that moment the whole process is a puppet show: the
timeline plays, the step counter climbs, sim time advances by exactly dt every
frame — and not one body is integrated.  A cube dropped from 3 m falls 0.0 m
over 4 s of "simulation".  One log line records any of this.

So the check has to happen BEFORE the geometry reaches the solver, which means
it has to run without the solver — no GPU, no USD, no Isaac.  That is what this
module is: the same rank test qhull performs, in pure Python, on a list of
vertices, so a scene builder can refuse to author a shape that will silently
kill physics.

Conventions, matching the rest of `geo/`:
  * Points are (x, y, z) in stage units; this module is unit-agnostic.
  * "Degenerate" means rank < 3 — every vertex lies on a common plane (or line,
    or point).  It is a property of the point SET, not of any axis, which is
    why an axis-aligned extent test alone is not enough (see the tilted-plane
    test below).
  * The strict entry point raises ValueError; the forgiving one returns a
    report, because a bridge wants to log and degrade, not crash.
"""

from __future__ import annotations

import pytest

from tritium_lib.geo.collider_shape import (
    ShapeReport,
    aabb_extent,
    check_convex_hull_input,
    validate_convex_hull_input,
)

# --- the shapes in play ------------------------------------------------------

# The exact point set qhull rejected on the RTX 4090, straight from the log.
FLAT_GROUND_PLANE = [
    (-50.0, -50.0, 0.0),
    (50.0, -50.0, 0.0),
    (50.0, 50.0, 0.0),
    (-50.0, 50.0, 0.0),
]

# The shape that replaced it: a 50 x 50 x 1 m slab.  Same footprint, real depth.
GROUND_SLAB = [
    (x, y, z)
    for x in (-25.0, 25.0)
    for y in (-25.0, 25.0)
    for z in (-1.0, 0.0)
]

UNIT_CUBE = [
    (x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)
]


# --- aabb_extent -------------------------------------------------------------


def test_aabb_extent_of_a_unit_cube_is_one_on_every_axis() -> None:
    """The trivial case, so the harder ones below have a known baseline."""
    assert aabb_extent(UNIT_CUBE) == pytest.approx((1.0, 1.0, 1.0))


def test_aabb_extent_reports_zero_depth_for_the_flat_ground_plane() -> None:
    """The z extent is the number that should have been noticed at authoring."""
    ex, ey, ez = aabb_extent(FLAT_GROUND_PLANE)
    assert ex == pytest.approx(100.0)
    assert ey == pytest.approx(100.0)
    assert ez == pytest.approx(0.0, abs=1e-9)


def test_aabb_extent_rejects_an_empty_point_set() -> None:
    """An empty mesh has no extent; returning zeros would read as 'flat'."""
    with pytest.raises(ValueError, match="points"):
        aabb_extent([])


# --- check_convex_hull_input: the degenerate cases ---------------------------


def test_the_flat_ground_plane_is_reported_as_not_hullable() -> None:
    """The regression itself: this exact quad must never reach a hull solver."""
    report = check_convex_hull_input(FLAT_GROUND_PLANE)
    assert isinstance(report, ShapeReport)
    assert report.is_hullable is False
    assert report.rank == 2
    assert "coplanar" in report.reason.lower()


def test_a_tilted_flat_quad_is_caught_even_though_no_axis_extent_is_zero() -> None:
    """The test that justifies doing rank properly instead of checking extents.

    Rotate the flat quad off-axis and every one of its three AABB extents is
    non-zero, so the cheap "does it have depth on all axes" screen passes it
    happily.  The point set is still rank 2 and qhull still fails.  A validator
    built on extents alone would wave this through and the sim would die in the
    same silent way, just harder to explain.
    """
    tilted = [(x, y, 0.25 * x + 0.1 * y) for (x, y, _z) in FLAT_GROUND_PLANE]
    ex, ey, ez = aabb_extent(tilted)
    assert ez > 1.0  # the naive screen sees real depth here

    report = check_convex_hull_input(tilted)
    assert report.is_hullable is False
    assert report.rank == 2


def test_collinear_points_are_rank_one() -> None:
    """A line of vertices — rank 1, and worth distinguishing from a plane."""
    line = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0)]
    report = check_convex_hull_input(line)
    assert report.is_hullable is False
    assert report.rank == 1


def test_identical_points_are_rank_zero() -> None:
    """Every vertex at the same place — a shape with no size at all."""
    report = check_convex_hull_input([(2.0, 2.0, 2.0)] * 5)
    assert report.is_hullable is False
    assert report.rank == 0


def test_fewer_than_four_points_cannot_bound_a_volume() -> None:
    """Three points are a triangle; a hull needs a tetrahedron's worth."""
    report = check_convex_hull_input(UNIT_CUBE[:3])
    assert report.is_hullable is False
    assert "4" in report.reason or "four" in report.reason.lower()


# --- check_convex_hull_input: the shapes that should pass --------------------


def test_the_unit_cube_is_hullable() -> None:
    """The everyday case must not be flagged, or the check is unusable."""
    report = check_convex_hull_input(UNIT_CUBE)
    assert report.is_hullable is True
    assert report.rank == 3
    assert report.reason == ""


def test_the_ground_slab_that_fixed_the_lane_is_hullable() -> None:
    """The actual replacement geometry, asserted as the shape that works."""
    report = check_convex_hull_input(GROUND_SLAB)
    assert report.is_hullable is True
    assert report.rank == 3


def test_a_thin_slab_is_hullable_when_its_depth_clears_the_tolerance() -> None:
    """Thin is fine; flat is not.  The boundary is the tolerance, not a guess.

    This matters because the instinct after being bitten is to demand a fat
    collider.  A 1 cm shell on a 100 m floor is legitimate geometry and must
    survive, so the check is a rank test with an explicit tolerance rather than
    a minimum-thickness rule.
    """
    slab = [
        (x, y, z)
        for x in (-50.0, 50.0)
        for y in (-50.0, 50.0)
        for z in (0.0, 0.01)
    ]
    assert check_convex_hull_input(slab, tolerance=1e-6).is_hullable is True


def test_tolerance_can_reject_a_shape_too_thin_for_a_given_solver() -> None:
    """Callers with a fussier backend can raise the bar and get a rejection."""
    slab = [
        (x, y, z)
        for x in (-50.0, 50.0)
        for y in (-50.0, 50.0)
        for z in (0.0, 0.001)
    ]
    assert check_convex_hull_input(slab, tolerance=0.01).is_hullable is False


@pytest.mark.parametrize("bad", [0.0, -1e-9, -1.0])
def test_rejects_a_non_positive_tolerance(bad: float) -> None:
    """A zero tolerance makes the rank test a coin-flip on floating-point noise."""
    with pytest.raises(ValueError, match="tolerance"):
        check_convex_hull_input(UNIT_CUBE, tolerance=bad)


# --- the report carries enough to act on -------------------------------------


def test_the_report_names_the_thinnest_direction_of_a_flat_shape() -> None:
    """A report that only says 'bad' leaves the author guessing at the fix.

    For the ground plane the useful sentence is "it has no depth along z" —
    that is the axis to give thickness to, and it is the plane normal, not
    whichever AABB extent happens to be smallest.
    """
    report = check_convex_hull_input(FLAT_GROUND_PLANE)
    assert report.plane_normal is not None
    nx, ny, nz = report.plane_normal
    assert abs(nx) == pytest.approx(0.0, abs=1e-9)
    assert abs(ny) == pytest.approx(0.0, abs=1e-9)
    assert abs(nz) == pytest.approx(1.0)


def test_a_hullable_shape_reports_no_plane_normal() -> None:
    """There is no degenerate plane to name when the shape is a real volume."""
    assert check_convex_hull_input(UNIT_CUBE).plane_normal is None


def test_the_report_carries_the_extent_so_a_caller_logs_one_object() -> None:
    """Callers should not have to call two functions to build one log line."""
    report = check_convex_hull_input(FLAT_GROUND_PLANE)
    assert report.extent == pytest.approx((100.0, 100.0, 0.0))


def test_check_never_raises_on_junk_input() -> None:
    """The forgiving half of the two-tier policy: bridges log, they don't crash."""
    assert check_convex_hull_input([]).is_hullable is False


# --- validate_convex_hull_input: the strict half -----------------------------


def test_validate_passes_a_real_volume_silently() -> None:
    """No return value, no exception — the shape is fine."""
    assert validate_convex_hull_input(UNIT_CUBE, name="/World/Cube") is None


def test_validate_raises_and_names_the_prim_for_the_flat_plane() -> None:
    """The message has to name the prim, or the author cannot find the shape.

    This is the whole point of the strict variant: turn a silent, delayed,
    process-wide physics death into a loud failure at the line that authored
    the bad geometry.
    """
    with pytest.raises(ValueError, match="/World/GroundPlane"):
        validate_convex_hull_input(FLAT_GROUND_PLANE, name="/World/GroundPlane")


def test_the_validate_message_explains_the_consequence() -> None:
    """A bare 'degenerate collider' teaches nobody what is about to happen."""
    with pytest.raises(ValueError) as excinfo:
        validate_convex_hull_input(FLAT_GROUND_PLANE, name="/World/GroundPlane")
    message = str(excinfo.value).lower()
    assert "coplanar" in message
    assert "hull" in message

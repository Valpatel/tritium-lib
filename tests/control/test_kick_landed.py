# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Did the push actually land ON THE AXIS IT WAS AIMED AT?

The disturbance module already guards the case where an impulse never fires.
This guards the subtler one that the first live 3 N-s A/B walked straight
into: a kick record *exists*, the solver *did* change the body's velocity, and
the run is scored as a legitimate trial -- but the velocity the body gained is
almost entirely on an axis the push never commanded.

Concretely, closed-loop trial 1 of that A/B recorded a lateral (+Y) 3 N-s push
and measured ``dv = [-0.089, 0.0121, 0.1921]``.  The *magnitude* of that vector
is 0.212 m/s, which is right about the expected ``J/m`` -- so any check on
magnitude alone calls it a clean push.  But the commanded axis is Y, and Y
gained 0.0121 m/s: about 6% of what was asked.  The body was falling (Z), not
shoved (Y).  That trial then counted a 178-degree tumble against the
closed-loop arm as a failure to reject a disturbance it was never given.

So the test that matters is a PROJECTION onto the commanded direction, not a
norm.  A norm here is actively worse than no check at all, because it passes
exactly the trials it needs to catch.
"""

import pytest

from tritium_lib.control import kick_landed


class TestProjectionNotMagnitude:
    """The discriminating case: right magnitude, wrong axis."""

    def test_falling_body_is_not_a_landed_push(self):
        # The real trial-1 numbers.  |dv| ~= expected, but Y did not move.
        verdict = kick_landed(
            commanded=(0.0, 3.0, 0.0),
            measured_dv=(-0.089, 0.0121, 0.1921),
            body_mass=15.0,
        )
        assert verdict.landed is False
        # And it must say WHY, or the exclusion is unauditable.
        assert verdict.fraction < 0.15

    def test_magnitude_alone_would_have_passed_that_trial(self):
        # Guard-rail on the guard-rail: assert the trap is real, so nobody
        # "simplifies" this back to a norm check later.
        dv = (-0.089, 0.0121, 0.1921)
        magnitude = sum(c * c for c in dv) ** 0.5
        expected = 3.0 / 15.0
        assert magnitude == pytest.approx(0.212, abs=0.01)
        assert magnitude > expected  # a norm check would call this a pass

    def test_clean_lateral_push_lands(self):
        # Trial 2 of the same A/B: Y gained 0.348 against an expected 0.2.
        verdict = kick_landed(
            commanded=(0.0, 3.0, 0.0),
            measured_dv=(0.0011, 0.3481, -0.0233),
            body_mass=15.0,
        )
        assert verdict.landed is True
        assert verdict.fraction > 1.0


class TestOverAndUnderDelivery:
    def test_exactly_expected_lands(self):
        v = kick_landed(commanded=(0.0, 3.0, 0.0),
                        measured_dv=(0.0, 0.2, 0.0), body_mass=15.0)
        assert v.landed is True
        assert v.fraction == pytest.approx(1.0)

    def test_partial_delivery_above_threshold_lands(self):
        # Foot friction absorbs a real share of a real push; tick 13 measured
        # about half of J/m and that was physically right.  Half must PASS.
        v = kick_landed(commanded=(0.0, 3.0, 0.0),
                        measured_dv=(0.0, 0.1, 0.0), body_mass=15.0)
        assert v.landed is True

    def test_below_threshold_does_not_land(self):
        v = kick_landed(commanded=(0.0, 3.0, 0.0),
                        measured_dv=(0.0, 0.05, 0.0), body_mass=15.0,
                        min_fraction=0.4)
        assert v.landed is False

    def test_threshold_is_inclusive_at_the_boundary(self):
        v = kick_landed(commanded=(0.0, 3.0, 0.0),
                        measured_dv=(0.0, 0.08, 0.0), body_mass=15.0,
                        min_fraction=0.4)
        assert v.landed is True

    def test_backwards_push_is_negative_and_never_lands(self):
        # Body moved OPPOSITE the commanded axis: that is not a weak push,
        # it is evidence something else drove the body.
        v = kick_landed(commanded=(0.0, 3.0, 0.0),
                        measured_dv=(0.0, -0.3, 0.0), body_mass=15.0)
        assert v.landed is False
        assert v.fraction < 0


class TestReportsItsOwnReasoning:
    def test_carries_expected_and_projected_for_the_record(self):
        v = kick_landed(commanded=(0.0, 3.0, 0.0),
                        measured_dv=(0.0, 0.1, 0.0), body_mass=15.0)
        assert v.expected_dv == pytest.approx(0.2)
        assert v.projected_dv == pytest.approx(0.1)
        d = v.as_dict()
        assert d["landed"] is True
        assert "fraction" in d and "projected_dv" in d

    def test_dict_is_json_safe(self):
        import json
        v = kick_landed(commanded=(0.0, 3.0, 0.0),
                        measured_dv=(0.0, 0.1, 0.0), body_mass=15.0)
        json.dumps(v.as_dict())


class TestRefusesToGuess:
    def test_zero_commanded_impulse_is_an_error(self):
        # Nothing was asked for, so "did it land" has no answer.  Returning
        # True here would silently validate an undisturbed run.
        with pytest.raises(ValueError, match="zero"):
            kick_landed(commanded=(0.0, 0.0, 0.0),
                        measured_dv=(0.0, 0.1, 0.0), body_mass=15.0)

    def test_nonpositive_mass_is_an_error(self):
        with pytest.raises(ValueError, match="mass"):
            kick_landed(commanded=(0.0, 3.0, 0.0),
                        measured_dv=(0.0, 0.1, 0.0), body_mass=0.0)

    def test_diagonal_push_projects_onto_its_own_direction(self):
        # A push aimed 45 degrees between X and Y, delivered exactly.
        import math
        j = 3.0
        cx = cy = j / math.sqrt(2)
        m = 15.0
        v = kick_landed(commanded=(cx, cy, 0.0),
                        measured_dv=(cx / m, cy / m, 0.0), body_mass=m)
        assert v.fraction == pytest.approx(1.0)
        assert v.landed is True

    def test_diagonal_push_delivered_sideways_does_not_land(self):
        # Same diagonal command, but the body moved perpendicular to it.
        import math
        j = 3.0
        cx = cy = j / math.sqrt(2)
        v = kick_landed(commanded=(cx, cy, 0.0),
                        measured_dv=(0.2, -0.2, 0.0), body_mass=15.0)
        assert v.fraction == pytest.approx(0.0, abs=1e-9)
        assert v.landed is False

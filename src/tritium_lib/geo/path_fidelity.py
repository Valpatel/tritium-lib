"""Compare where a body actually went against where the map says it went.

Every layer of the sensing chain can be individually healthy while the
picture the operator sees is wrong.  A pose ingest can wedge and keep
repainting its first fix; a frame conversion can transpose two axes; a
buffered feed can be two seconds behind the world.  In all three cases the
track *exists*, the endpoint returns 200, and the tactical map looks alive.
The only check that catches them is comparing the reported path against
ground truth, sample by sample, in time.

That is what this module does, and it is deliberately body-agnostic: the
reference can be an Isaac root transform, a motion-capture rig, or RTK GPS,
and the report can be a fused track, a raw sighting stream, or a replay.
Coordinates are plain metres in a shared local frame -- converting lat/lon
into that frame is the caller's job, because doing it here would bake in a
datum this module has no business choosing.

Two properties make the verdict hard to fool:

* **Time alignment.** The reference is linearly interpolated at each reported
  timestamp, so a track with perfect shape but a lag still scores as error.
  A shape-only comparison calls a two-second-late replay a perfect match.
* **Max, not mean.** The verdict keys off the worst sample.  A single large
  excursion in an otherwise clean track is a real failure, and averaging is
  exactly the operation that hides it.

Stdlib only -- no numpy -- so it imports on a bare Jetson and inside a
simulator's embedded Python alike.

Created by Matthew Valancy / Copyright 2026 Valpatel Software LLC / AGPL-3.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "DEFAULT_DIVERGED_M",
    "DEFAULT_TOLERANCE_M",
    "PathFidelity",
    "PathSample",
    "compare_paths",
    "heading_error_deg",
    "path_length_m",
]

#: Below this worst-sample error the two paths are called the same path.
DEFAULT_TOLERANCE_M = 0.50

#: A reported path shorter than this fraction of the reference is treated as
#: a different path rather than a noisy one, however small its absolute
#: error.  Healthy tracks sit near 1.0; noise nudges the ratio up, not down.
MIN_LENGTH_RATIO = 0.50

#: At or above this worst-sample error the report is not a degraded version
#: of the truth, it is a different path -- a sign flip, an axis swap, or a
#: dead ingest rather than noise.
DEFAULT_DIVERGED_M = 2.00


@dataclass(frozen=True)
class PathSample:
    """One timestamped planar pose.

    ``t`` is seconds on any clock shared by both paths (an epoch or a
    seconds-since-start both work, as long as the two agree).  ``heading_deg``
    is optional and follows whatever convention the caller uses on both
    sides; it is only ever compared to another heading, never interpreted.
    """

    t: float
    x: float
    y: float
    heading_deg: float | None = None


@dataclass(frozen=True)
class PathFidelity:
    """The scored comparison of a reported path against a reference path."""

    verdict: str
    samples_compared: int
    max_error_m: float | None
    rms_error_m: float | None
    mean_error_m: float | None
    final_error_m: float | None
    reference_length_m: float
    reported_length_m: float
    length_ratio: float | None
    max_heading_error_deg: float | None
    mean_heading_error_deg: float | None
    heading_samples_compared: int
    overlap_s: float
    tolerance_m: float
    diverged_m: float

    def as_dict(self) -> dict:
        """A JSON-safe dict, for dropping straight into a run report."""
        return {
            "verdict": self.verdict,
            "samples_compared": self.samples_compared,
            "max_error_m": _round(self.max_error_m, 4),
            "rms_error_m": _round(self.rms_error_m, 4),
            "mean_error_m": _round(self.mean_error_m, 4),
            "final_error_m": _round(self.final_error_m, 4),
            "reference_length_m": _round(self.reference_length_m, 4),
            "reported_length_m": _round(self.reported_length_m, 4),
            "length_ratio": _round(self.length_ratio, 4),
            "max_heading_error_deg": _round(self.max_heading_error_deg, 3),
            "mean_heading_error_deg": _round(self.mean_heading_error_deg, 3),
            "heading_samples_compared": self.heading_samples_compared,
            "overlap_s": _round(self.overlap_s, 3),
            "tolerance_m": self.tolerance_m,
            "diverged_m": self.diverged_m,
        }

    def summary(self) -> str:
        """One line for a log or a commit message."""
        if not self.samples_compared:
            return f"{self.verdict}: 0 samples compared"
        heading = (
            f", heading max {self.max_heading_error_deg:.2f} deg"
            if self.max_heading_error_deg is not None
            else ""
        )
        ratio = (
            f", length ratio {self.length_ratio:.3f}"
            if self.length_ratio is not None
            else ""
        )
        return (
            f"{self.verdict}: {self.samples_compared} samples, "
            f"max {self.max_error_m:.4f} m, rms {self.rms_error_m:.4f} m, "
            f"final {self.final_error_m:.4f} m{ratio}{heading}"
        )


def _round(value: float | None, places: int) -> float | None:
    return None if value is None else round(value, places)


def path_length_m(samples: Iterable[PathSample]) -> float:
    """Arc length walked, not displacement.

    An out-and-back run has zero displacement and a real length; conflating
    the two is how a pacing robot gets scored as motionless.
    """
    ordered = sorted(samples, key=lambda s: s.t)
    return sum(
        math.hypot(b.x - a.x, b.y - a.y)
        for a, b in zip(ordered, ordered[1:])
    )


def heading_error_deg(a_deg: float, b_deg: float) -> float:
    """Absolute angular difference, wrapped the short way around the circle.

    359 deg and 1 deg differ by 2, not 358.  Without the wrap, a track
    straddling north reports a near-maximal error every time it crosses.
    """
    return abs((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def _interpolate(reference: Sequence[PathSample], t: float) -> PathSample | None:
    """The reference pose at time ``t``, or None if ``t`` is outside its span.

    Outside the span the honest answer is "unknown".  Clamping to the nearest
    endpoint would silently compare a reported sample against a pose the
    reference never claimed, which manufactures agreement at the head of a
    track and error at the tail.
    """
    if t < reference[0].t or t > reference[-1].t:
        return None

    lo, hi = 0, len(reference) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if reference[mid].t < t:
            lo = mid + 1
        else:
            hi = mid
    exact = reference[lo]
    if exact.t == t or lo == 0:
        return exact

    before, after = reference[lo - 1], exact
    span = after.t - before.t
    frac = 0.0 if span <= 0 else (t - before.t) / span
    heading = None
    if before.heading_deg is not None and after.heading_deg is not None:
        # Interpolate along the short arc so a segment spanning north does
        # not sweep the long way around.
        delta = (after.heading_deg - before.heading_deg + 180.0) % 360.0 - 180.0
        heading = (before.heading_deg + delta * frac) % 360.0
    return PathSample(
        t=t,
        x=before.x + (after.x - before.x) * frac,
        y=before.y + (after.y - before.y) * frac,
        heading_deg=heading,
    )


def compare_paths(
    reference: Sequence[PathSample],
    reported: Sequence[PathSample],
    tolerance_m: float = DEFAULT_TOLERANCE_M,
    diverged_m: float = DEFAULT_DIVERGED_M,
) -> PathFidelity:
    """Score ``reported`` against ground-truth ``reference``.

    Both are sorted by time here rather than trusted to arrive ordered -- a
    sighting stream that reorders under load is a normal thing, not a caller
    error.  Reported samples outside the reference's time span are dropped.

    Raises ``ValueError`` if the reference is empty: with no ground truth
    there is no comparison to make, and returning a passing verdict for
    "nothing to check against" is the exact failure this module exists to
    prevent.  An empty *report* is fine, and scores ``NO_OVERLAP`` -- a body
    that walked and produced no track at all is a real, reportable outcome.
    """
    ref = sorted(reference, key=lambda s: s.t)
    rep = sorted(reported, key=lambda s: s.t)
    if not ref:
        raise ValueError("reference path is empty -- nothing to compare against")

    ref_length = path_length_m(ref)
    rep_length = path_length_m(rep)
    # A stationary reference has no scale to normalise by; a ratio against it
    # is either a divide-by-zero or an infinity that reads as disagreement.
    length_ratio = (rep_length / ref_length) if ref_length > 0 else None

    errors: list[float] = []
    heading_errors: list[float] = []
    matched_times: list[float] = []
    for sample in rep:
        truth = _interpolate(ref, sample.t)
        if truth is None:
            continue
        errors.append(math.hypot(sample.x - truth.x, sample.y - truth.y))
        matched_times.append(sample.t)
        if sample.heading_deg is not None and truth.heading_deg is not None:
            heading_errors.append(
                heading_error_deg(sample.heading_deg, truth.heading_deg)
            )

    if not errors:
        return PathFidelity(
            verdict="NO_OVERLAP",
            samples_compared=0,
            max_error_m=None,
            rms_error_m=None,
            mean_error_m=None,
            final_error_m=None,
            reference_length_m=ref_length,
            reported_length_m=rep_length,
            length_ratio=length_ratio,
            max_heading_error_deg=None,
            mean_heading_error_deg=None,
            heading_samples_compared=0,
            overlap_s=0.0,
            tolerance_m=tolerance_m,
            diverged_m=diverged_m,
        )

    max_error = max(errors)
    mean_error = sum(errors) / len(errors)
    rms_error = math.sqrt(sum(e * e for e in errors) / len(errors))

    # The verdict keys off the worst sample on purpose.  Averaging is the
    # operation that hides a single large excursion, and a single large
    # excursion is a real defect.
    if max_error < tolerance_m:
        verdict = "AGREES"
    elif max_error < diverged_m:
        verdict = "DRIFTS"
    else:
        verdict = "DIVERGED"

    # Shape guard.  Absolute error alone cannot see a wedged or stalling
    # ingest during a SHORT walk: freeze the track at the start of a 1.2 m
    # stroll and the worst sample is still only ~0.4 m out, comfortably
    # inside a half-metre tolerance, so the verdict comes back AGREES while
    # the reported path covers a sixth of the ground the body did.  This is
    # not hypothetical -- it is what the live frozen-ingest control actually
    # produced, and it is the reason this check exists.
    #
    # A healthy track sits near ratio 1.0 (noise pushes it slightly above,
    # never far below).  Covering less than half the truth is a different
    # path, not a noisier one.  The reference must itself have moved
    # meaningfully first: with nothing to be short of, a frozen report is
    # legitimately correct and inventing a failure here would be worse than
    # the hole it closes.
    if (
        length_ratio is not None
        and length_ratio < MIN_LENGTH_RATIO
        and ref_length > 2.0 * tolerance_m
    ):
        verdict = "DIVERGED"

    return PathFidelity(
        verdict=verdict,
        samples_compared=len(errors),
        max_error_m=max_error,
        rms_error_m=rms_error,
        mean_error_m=mean_error,
        final_error_m=errors[-1],
        reference_length_m=ref_length,
        reported_length_m=rep_length,
        length_ratio=length_ratio,
        max_heading_error_deg=max(heading_errors) if heading_errors else None,
        mean_heading_error_deg=(
            sum(heading_errors) / len(heading_errors) if heading_errors else None
        ),
        heading_samples_compared=len(heading_errors),
        overlap_s=matched_times[-1] - matched_times[0],
        tolerance_m=tolerance_m,
        diverged_m=diverged_m,
    )

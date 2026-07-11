# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Perception helpers — seasonal concealment -> detectability.

The GIS lane exposes a per-point *concealment* term (0..1) that already folds
in the season: a hostile standing in a summer forest is under full canopy
(concealment ~0.81) while the same stand bare in winter barely hides it
(~0.12).  See ``tritium_lib.geo.gis.landcover.LandCoverGrid.concealment_at``.

This module is the small, reusable bridge from that spatial/seasonal
concealment to a *combat detectability* term the perception/acquisition path
consumes.  It deliberately holds NO combat state and NO GIS import — the land
cover grid is duck-typed (anything exposing
``concealment_at(lat, lon, foliage=...) -> float``), so combat behaviour, the
vision system, or an addon can reuse it without dragging in geo dependencies.

Contract (the byte-identical guarantee combat relies on):
    detectability_multiplier(0.0) == 1.0   # exposed target — no change
and ``base_range * 1.0 == base_range`` in IEEE-754, so a scenario that never
attaches a concealment field acquires targets exactly as before.
"""

from __future__ import annotations

# A concealed target is never made *perfectly* invisible — a unit standing on
# top of it can still acquire.  Mirrors the acquisition floor used by the
# global weather detection modifier (UnitBehaviors.set_detection_modifier), so
# the two degradations compose without either one blinding a unit completely.
CONCEALMENT_FLOOR = 0.05


def detectability_multiplier(concealment: float) -> float:
    """Map seasonal concealment (0..1) to an acquisition-range multiplier.

    ``concealment`` is the GIS per-point value: 0.0 fully exposed, ~0.8 a
    summer forest canopy.  The returned multiplier scales the range at which an
    observer can acquire (open fire on) a target standing there:

        effective_range = base_range * detectability_multiplier(concealment)

    A fully exposed target (``concealment == 0.0``) returns exactly ``1.0`` so
    the acquisition range is unchanged bit-for-bit (the gating no-op).  A summer
    forest (``0.81``) returns ``0.19`` — the observer must be ~5x closer.  The
    result is clamped to ``CONCEALMENT_FLOOR`` so a target is never perfectly
    invisible, and to ``1.0`` at the top so an out-of-range/negative concealment
    never *extends* acquisition.
    """
    c = float(concealment)
    if c <= 0.0:
        return 1.0
    if c >= 1.0:
        return CONCEALMENT_FLOOR
    m = 1.0 - c
    return CONCEALMENT_FLOOR if m < CONCEALMENT_FLOOR else m


class ConcealmentField:
    """Bind a land-cover grid + live foliage + a coordinate mapper into a
    ``target -> concealment`` callable the acquisition path can query.

    This is the reusable glue between the GIS/environment lanes and combat
    perception, with zero combat or GIS *type* coupling:

    * ``grid`` — anything exposing ``concealment_at(lat, lon, foliage=...)``
      (a :class:`~tritium_lib.geo.gis.landcover.LandCoverGrid` in practice).
    * ``foliage_fn`` — ``() -> float`` sampled *per query* so the season is
      always current (e.g. ``Environment.foliage_state``).  ``None`` -> 1.0
      (summer / full canopy).
    * ``to_lonlat`` — ``(x, y) -> (lon, lat)`` mapping sim world metres to the
      grid's WGS-84 frame.  ``None`` treats ``target.position`` as ``(lon, lat)``
      already (useful for tests and geo-native callers).

    Calling the field with a target returns its concealment in ``0..1``; an
    out-of-grid or NoData point yields ``0.0`` (fully exposed — the grid's own
    safe default).
    """

    def __init__(self, grid, foliage_fn=None, to_lonlat=None) -> None:
        self._grid = grid
        self._foliage_fn = foliage_fn
        self._to_lonlat = to_lonlat

    def set_foliage_fn(self, foliage_fn) -> None:
        """Rebind the live seasonal foliage source (``() -> float`` or None).

        The grid + coordinate mapping are stable across a shift, but a caller
        that caches the field can swap in a fresh foliage source without
        rebuilding — the season is always read live at query time.
        """
        self._foliage_fn = foliage_fn

    def foliage(self) -> float:
        """Current seasonal foliage (1.0 when no season source attached)."""
        if self._foliage_fn is None:
            return 1.0
        try:
            return float(self._foliage_fn())
        except Exception:
            return 1.0

    def concealment_for(self, target) -> float:
        """Seasonal concealment (0..1) for ``target`` at its current position."""
        if self._grid is None:
            return 0.0
        x, y = target.position[0], target.position[1]
        if self._to_lonlat is not None:
            lon, lat = self._to_lonlat(x, y)
        else:
            lon, lat = x, y
        try:
            return self._grid.concealment_at(lat, lon, foliage=self.foliage())
        except Exception:
            return 0.0

    def __call__(self, target) -> float:
        return self.concealment_for(target)

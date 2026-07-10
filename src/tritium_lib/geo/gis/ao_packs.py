# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Area-of-Operations (AO) pack registry — derived from packaged fixtures.

Tritium is **not tied to a single Area of Operations**.  Two real fixture packs
ship in :mod:`tritium_lib.geo.gis.fixtures` (see that package's docstring):

    * **Dublin, CA** — the original demo AO (``*_ao.json``).
    * **Boulder, CO** — a second real AO with strong mountains-to-plains relief
      (``*_boulder.json``).  No ``noaa_alerts_boulder.json`` exists because there
      were no active NWS alerts over the AO at capture time (a legitimately
      empty layer).

This module turns those on-disk packs into a small, reusable registry an
operator surface (or a real deployment standing up at a new site) can pick from
— *pick your battlefield*.  Every field is **derived from the fixture files
themselves**:

    * the AO bounding box comes from each pack's DEM fixture — the top-level
      ``"bbox": [w, s, e, n]`` marker written by the capture tool, falling back
      to the DEM grid's own ``west/south/east/north`` fields (the original
      Dublin DEM predates the marker).  This mirrors
      :meth:`tritium_lib.geo.gis.fetchers.UsgsElevationFetcher._fixture_bbox`
      so a box is never hard-coded twice.
    * the ``layers`` list reflects exactly which ``{stem}_{suffix}.json`` files
      are present for that AO, so Boulder legitimately reports no
      ``noaa_alerts``.

Graceful degradation: an AO with **no** packaged fixtures is dropped, so on a
build where the fixtures package is absent :func:`list_ao_packs` returns ``[]``
and the operator surface simply hides its AO picker.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import resources

logger = logging.getLogger(__name__)

_FIXTURE_PKG = "tritium_lib.geo.gis.fixtures"

#: AO definitions: ``(id, display name, fixture filename suffix)``.  The suffix
#: is the ``{stem}_{suffix}.json`` tail the capture tool wrote for each pack.
_AO_DEFS: tuple[tuple[str, str, str], ...] = (
    ("dublin", "Dublin, CA", "ao"),
    ("boulder", "Boulder, CO", "boulder"),
)

#: Fixture data-layer stems, in roughly sensor-priority order.  ``layers``
#: reports which of these packs exist for an AO (so Boulder legitimately omits
#: ``noaa_alerts``).  The DEM stem is also the AO bounding-box source.
_LAYER_STEMS: tuple[str, ...] = (
    "usgs_dem",
    "tiger_roads",
    "fema_flood",
    "osm_buildings",
    "noaa_alerts",
)

_DEM_STEM = "usgs_dem"

__all__ = ["AOPack", "list_ao_packs", "get_ao_pack", "active_ao_id"]


@dataclass(frozen=True)
class AOPack:
    """One Area of Operations, derived from its packaged fixtures.

    Attributes:
        id: Stable pack id (``"dublin"`` / ``"boulder"``).
        name: Human-readable display name.
        bbox: ``(west, south, east, north)`` in WGS-84 degrees.
        center_lat: Latitude of the bbox center (the geo-reference target).
        center_lng: Longitude of the bbox center.
        layers: Fixture data-layer stems packaged for this AO.
    """

    id: str
    name: str
    bbox: tuple[float, float, float, float]
    center_lat: float
    center_lng: float
    layers: tuple[str, ...]

    def contains(self, lat: float, lng: float) -> bool:
        """True when ``(lat, lng)`` falls inside this AO's bbox (edges count)."""
        w, s, e, n = self.bbox
        return (s <= lat <= n) and (w <= lng <= e)

    def to_dict(self) -> dict:
        """JSON-serializable dict for the ``/api/gis/ao`` route + frontend."""
        w, s, e, n = self.bbox
        return {
            "id": self.id,
            "name": self.name,
            "bbox": [w, s, e, n],
            "center": {"lat": self.center_lat, "lng": self.center_lng},
            "layers": list(self.layers),
        }


def _fixture_name(stem: str, suffix: str) -> str:
    return f"{stem}_{suffix}.json"


def _load_fixture(name: str):
    """Load a packaged fixture JSON by filename, or ``None`` if absent/bad."""
    try:
        resource = resources.files(_FIXTURE_PKG).joinpath(name)
        return json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError, OSError, ValueError) as exc:
        logger.debug("AO fixture %s unavailable: %s", name, exc)
        return None


def _fixture_exists(name: str) -> bool:
    """True when the packaged fixture ``name`` is present."""
    try:
        return resources.files(_FIXTURE_PKG).joinpath(name).is_file()
    except (FileNotFoundError, ModuleNotFoundError, OSError, ValueError):
        return False


def _bbox_from_fixture(fixture):
    """AO bbox ``(w, s, e, n)`` from a fixture dict, or ``None``.

    Prefers the top-level ``"bbox": [w, s, e, n]`` marker (written by the AO
    capture tool); falls back to the DEM grid's ``west/south/east/north`` fields
    (the original Dublin DEM predates the marker).
    """
    if not isinstance(fixture, dict):
        return None
    marker = fixture.get("bbox")
    if isinstance(marker, (list, tuple)) and len(marker) == 4:
        try:
            return (float(marker[0]), float(marker[1]),
                    float(marker[2]), float(marker[3]))
        except (TypeError, ValueError):
            pass
    try:
        return (float(fixture["west"]), float(fixture["south"]),
                float(fixture["east"]), float(fixture["north"]))
    except (KeyError, TypeError, ValueError):
        return None


def _pack_for(ao_id: str, name: str, suffix: str):
    """Build an :class:`AOPack` for one AO def, or ``None`` if it has no data."""
    layers = tuple(
        stem for stem in _LAYER_STEMS
        if _fixture_exists(_fixture_name(stem, suffix))
    )
    if not layers:
        return None  # no packaged data for this AO — drop it

    # Derive the bbox from the DEM fixture (both AOs ship one with w/s/e/n);
    # fall back to any present fixture that carries a usable bbox marker.
    bbox = None
    dem = _load_fixture(_fixture_name(_DEM_STEM, suffix))
    if dem is not None:
        bbox = _bbox_from_fixture(dem)
    if bbox is None:
        for stem in layers:
            fixture = _load_fixture(_fixture_name(stem, suffix))
            bbox = _bbox_from_fixture(fixture)
            if bbox is not None:
                break
    if bbox is None:
        return None

    w, s, e, n = bbox
    return AOPack(
        id=ao_id,
        name=name,
        bbox=(w, s, e, n),
        center_lat=(s + n) / 2.0,
        center_lng=(w + e) / 2.0,
        layers=layers,
    )


def list_ao_packs() -> list[AOPack]:
    """Return every AO that has packaged fixtures, in registry order.

    Empty when the fixtures package is absent — the caller then hides its AO
    picker (graceful degradation).
    """
    packs: list[AOPack] = []
    for ao_id, name, suffix in _AO_DEFS:
        pack = _pack_for(ao_id, name, suffix)
        if pack is not None:
            packs.append(pack)
    return packs


def get_ao_pack(ao_id: str):
    """Return the AO pack with ``ao_id``, or ``None`` if unknown."""
    for pack in list_ao_packs():
        if pack.id == ao_id:
            return pack
    return None


def active_ao_id(lat: float, lng: float):
    """Return the id of the first AO whose bbox contains ``(lat, lng)``, else ``None``."""
    for pack in list_ao_packs():
        if pack.contains(lat, lng):
            return pack.id
    return None

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Reusable Area-of-Operations (AO) fixture capture tool.

The packaged offline fixtures in ``fixtures/`` are **not Dublin-specific**: this
module runs the live fetchers over *any* bounding box and writes a matching set
of trimmed, coordinate-rounded fixture files — which is what proves (and
enables) multi-AO offline support.  It is how the Boulder, CO pack was produced
and how a third AO would be.

Pure stdlib.  Be polite: one capture run hits each public API once.  The DEM is
always sampled at :data:`DEM_NCOLS` x :data:`DEM_NROWS` (small: polite + a
compact fixture).  Every written file carries the ``"fixture": true`` marker and
a top-level ``"bbox": [w, s, e, n]`` (the AO box) so the fetchers'
intersection/clip checks stay cheap.  A source whose live fetch fails or returns
zero features is **skipped and reported** — never fatal for the rest of the run.

Example::

    from tritium_lib.geo.gis.capture import capture_ao_pack
    summary = capture_ao_pack(
        bbox="-105.30,39.98,-105.26,40.02", name="boulder",
        out_dir="src/tritium_lib/geo/gis/fixtures",
    )
    # {'tiger_roads_boulder.json': 150, 'usgs_dem_boulder.json': '32x32', ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .fetchers import (
    FemaFloodFetcher,
    NoaaAlertsFetcher,
    OverpassBuildingsFetcher,
    TigerRoadsFetcher,
    UsgsElevationFetcher,
    _as_bbox,
)

__all__ = ["capture_ao_pack", "default_fetchers", "DEM_NCOLS", "DEM_NROWS"]

logger = logging.getLogger(__name__)

#: DEM sample grid for a captured pack (kept small: polite + compact fixture).
DEM_NCOLS = 32
DEM_NROWS = 32
#: Elevation values are rounded to this many decimals in the written fixture —
#: centimetre precision is ample for terrain and keeps a 32x32 grid compact.
DEM_VALUE_PRECISION = 2


def default_fetchers() -> list:
    """The standard AO fetcher set, each built cache-free with a generous
    timeout for a one-shot live capture (Overpass is the slowest)."""
    return [
        TigerRoadsFetcher(cache=None, timeout_s=60.0),
        FemaFloodFetcher(cache=None, timeout_s=60.0),
        NoaaAlertsFetcher(cache=None, timeout_s=30.0),
        OverpassBuildingsFetcher(cache=None, timeout_s=90.0),
        UsgsElevationFetcher(cache=None, timeout_s=60.0),
    ]


def _stem_for(fetcher) -> str:
    """Filename stem for a fetcher, derived from its Dublin ``FIXTURE_NAME``.

    ``"tiger_roads_ao.json"`` -> ``"tiger_roads"`` (so a Boulder capture writes
    ``tiger_roads_boulder.json``).  Falls back to ``SOURCE`` then ``"layer"``.
    """
    stem = getattr(fetcher, "FIXTURE_NAME", "") or ""
    if stem.endswith(".json"):
        stem = stem[:-5]
    if stem.endswith("_ao"):
        stem = stem[:-3]
    if not stem:
        stem = getattr(fetcher, "SOURCE", "") or "layer"
    return stem


def _round_coords(coords, precision: int):
    """Recursively round every float in a GeoJSON coordinate tree.

    A leaf position (``[lon, lat]`` / ``[lon, lat, z]``) is detected by its two
    leading numbers — the same heuristic the fetchers use to walk geometry — so
    every position at any nesting depth is rounded.
    """
    if isinstance(coords, (list, tuple)):
        if (
            len(coords) >= 2
            and isinstance(coords[0], (int, float))
            and isinstance(coords[1], (int, float))
        ):
            return [round(float(c), precision) for c in coords]
        return [_round_coords(child, precision) for child in coords]
    return coords


def _round_feature(feature: dict, precision: int) -> dict:
    """Return a shallow copy of *feature* with its geometry coordinates rounded."""
    geom = dict((feature or {}).get("geometry") or {})
    if "coordinates" in geom:
        geom["coordinates"] = _round_coords(geom["coordinates"], precision)
    out = dict(feature or {})
    out["geometry"] = geom
    return out


def _write_json(path: Path, payload) -> int:
    """Write *payload* compactly (matching the checked-in fixtures). Returns bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _capture_vector(fetcher, box, path: Path, ao_bbox, max_features: int, precision: int):
    """Capture one vector fetcher to *path*.  Returns a feature count or a
    ``"skipped: ..."`` reason string (never raises for one bad source)."""
    try:
        fc = fetcher.fetch(box)
    except Exception as exc:  # noqa: BLE001 - one bad source never aborts the run
        logger.warning("capture: %s live fetch failed: %s", path.name, exc)
        return f"skipped: fetch failed ({exc})"
    features = (fc or {}).get("features") or []
    if not features:
        # Empty means either a genuine no-data AO or a failed live fetch that
        # degraded to a fixture that clips to nothing over this bbox — either
        # way there is nothing real to package, so skip (don't write the file).
        return "skipped: 0 features"
    capped = [_round_feature(f, precision) for f in features[:max_features]]
    payload = {
        "type": "FeatureCollection",
        "fixture": True,
        "bbox": ao_bbox,
        "features": capped,
    }
    _write_json(path, payload)
    return len(capped)


def _capture_dem(fetcher, box, path: Path, ao_bbox, precision: int):
    """Capture the elevation fetcher to *path* as a DEM fixture.  Returns a
    ``"COLSxROWS"`` shape string or a ``"skipped: ..."`` reason string."""
    try:
        grid = fetcher.fetch_grid(box, ncols=DEM_NCOLS, nrows=DEM_NROWS)
    except Exception as exc:  # noqa: BLE001 - never fatal for the rest of the run
        logger.warning("capture: %s live fetch failed: %s", path.name, exc)
        return f"skipped: fetch failed ({exc})"
    mn, _mx = grid.min_max()
    if mn is None:
        # All-NoData: a live failure that degraded to the empty grid (no
        # packaged AO intersects this bbox).  Nothing real to package.
        return "skipped: all-NoData grid"
    data = grid.to_dict()
    for corner in ("west", "south", "east", "north"):
        data[corner] = round(float(data[corner]), precision)
    data["values"] = [
        None if v is None else round(float(v), DEM_VALUE_PRECISION)
        for v in data.get("values", [])
    ]
    src = data.get("source") or "usgs"
    if not src.endswith("-fixture"):
        src = f"{src}-fixture"      # mark a packaged fixture, matching Dublin
    data["source"] = src
    data["fixture"] = True
    data["bbox"] = ao_bbox
    _write_json(path, data)
    return f"{grid.ncols}x{grid.nrows}"


def capture_ao_pack(
    bbox,
    name: str,
    out_dir,
    fetchers=None,
    max_features: int = 150,
    precision: int = 6,
) -> dict:
    """Capture a full offline fixture pack for one Area of Operations.

    Runs each fetcher live over *bbox*, caps feature counts to *max_features*,
    rounds every coordinate to *precision* decimals, and writes compact
    ``{stem}_{name}.json`` files (e.g. ``tiger_roads_boulder.json``) into
    *out_dir*.  Each written file is stamped ``"fixture": true`` and carries the
    AO box as a top-level ``"bbox": [w, s, e, n]`` so the fetchers' clip /
    intersection checks are cheap.  The DEM is sampled at
    :data:`DEM_NCOLS` x :data:`DEM_NROWS` and written from
    ``ElevationGrid.to_dict()`` plus the fixture / bbox markers, with its source
    tag suffixed ``-fixture``.

    Parameters
    ----------
    bbox : GeoBBox | str | tuple
        The AO bounding box, ``west, south, east, north``.
    name : str
        AO slug used in filenames (e.g. ``"boulder"``).
    out_dir : str | os.PathLike
        Destination directory (created if needed).
    fetchers : iterable, optional
        Fetcher instances to run.  Defaults to :func:`default_fetchers` (all
        five real sources).  A vector fetcher is any object with ``fetch``; the
        DEM fetcher is detected by having ``fetch_grid``.
    max_features, precision : int
        Feature cap and coordinate rounding.

    Returns
    -------
    dict
        ``{filename: value}`` where *value* is a feature count (vector), a
        ``"COLSxROWS"`` shape (DEM), or a ``"skipped: <reason>"`` string.  A
        source whose live fetch fails or returns zero features is skipped and
        reported — never raised.  Pure stdlib.
    """
    box = _as_bbox(bbox)
    out = Path(out_dir)
    if fetchers is None:
        fetchers = default_fetchers()
    ao_bbox = [
        round(box.west, precision),
        round(box.south, precision),
        round(box.east, precision),
        round(box.north, precision),
    ]
    summary: dict = {}
    for fetcher in fetchers:
        filename = f"{_stem_for(fetcher)}_{name}.json"
        path = out / filename
        if hasattr(fetcher, "fetch_grid"):
            summary[filename] = _capture_dem(fetcher, box, path, ao_bbox, precision)
        else:
            summary[filename] = _capture_vector(
                fetcher, box, path, ao_bbox, max_features, precision
            )
    return summary

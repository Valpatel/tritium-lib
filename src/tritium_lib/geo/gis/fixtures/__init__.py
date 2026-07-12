# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Packaged multi-AO GIS fixtures (offline fallback for the fetchers).

These JSON files are the offline fallback for the fetchers in the parent
package.  The system is **not tied to one Area of Operations**: each fetcher
declares an ordered ``FIXTURE_NAMES`` tuple and, offline, returns the first pack
whose data covers the requested bbox.

Two real packs ship today, named ``{layer}_{ao}.json``:

    * **Dublin, CA** (``*_ao.json``) — the original demo AO
      (bbox ``-121.912,37.704,-121.880,37.728``).
    * **Boulder, CO** (``*_boulder.json``) — a second real AO with strong
      mountains-to-plains relief (bbox ``-105.30,39.98,-105.26,40.02``),
      captured with :func:`tritium_lib.geo.gis.capture.capture_ao_pack`.
      No ``noaa_alerts_boulder.json`` exists: there were no active NWS alerts
      over the AO at capture time (a legitimately-empty layer).

Every fixture carries a top-level ``"fixture": true`` marker; packs written by
the capture tool also carry a top-level ``"bbox": [w, s, e, n]`` (the AO box) so
the fetchers' clip / intersection checks are cheap.  Each was produced by
running the fetcher ``parse_*`` functions over real captured government payloads,
then trimmed and coordinate-rounded.  Loaded via ``importlib.resources`` — do not
read them by hand-built path.
"""

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Packaged demo-AO GIS fixtures (Dublin, CA).

These JSON files are the offline fallback for the fetchers in the parent
package.  Each was produced by running the fetcher ``parse_*`` functions over
real captured government payloads, then trimmed and coordinate-rounded.  Every
fixture carries a top-level ``"fixture": true`` marker.  Loaded via
``importlib.resources`` — do not read them by hand-built path.
"""

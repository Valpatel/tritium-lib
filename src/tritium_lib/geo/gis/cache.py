# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""On-disk JSON cache for fetched GIS layers.

The cache lets every provider degrade gracefully: a live fetch that succeeds is
written here, so a later fetch that cannot reach the network can still serve the
last good answer (with no age limit) before falling back to the packaged demo
fixtures.  All IO is best-effort — a broken cache never raises, it just misses.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

__all__ = ["GISCache"]

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "data/gis_cache"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class GISCache:
    """A tiny filename-keyed JSON cache under ``cache_dir``."""

    def __init__(self, cache_dir: str | os.PathLike | None = None):
        if cache_dir is None:
            cache_dir = os.environ.get("TRITIUM_GIS_CACHE", _DEFAULT_DIR)
        self.cache_dir = Path(cache_dir)

    def key(self, source: str, bbox, **params) -> str:
        """Build a filename-safe cache key.

        ``bbox`` may be a :class:`GeoBBox` (anything with ``west/south/east/
        north``) or a 4-tuple ``(w, s, e, n)``.  Coordinates are rounded to 4
        decimal places so near-identical viewports share a cache entry.
        """
        w, s, e, n = self._bbox_tuple(bbox)
        parts = [
            str(source),
            f"{w:.4f}",
            f"{s:.4f}",
            f"{e:.4f}",
            f"{n:.4f}",
        ]
        for name in sorted(params):
            parts.append(f"{name}-{params[name]}")
        raw = "_".join(parts)
        return _UNSAFE.sub("-", raw)

    @staticmethod
    def _bbox_tuple(bbox) -> tuple:
        if hasattr(bbox, "west"):
            return (bbox.west, bbox.south, bbox.east, bbox.north)
        w, s, e, n = bbox
        return (float(w), float(s), float(e), float(n))

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str, max_age_s: float | None = None):
        """Return the cached payload, or ``None`` on miss / expiry / corruption.

        ``max_age_s=None`` means "no age limit" — any readable entry is valid.
        """
        path = self._path(key)
        try:
            if not path.is_file():
                return None
            if max_age_s is not None:
                age = time.time() - path.stat().st_mtime
                if age > max_age_s:
                    return None
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError) as exc:
            logger.debug("GIS cache read failed for %s: %s", key, exc)
            return None

    def put(self, key: str, payload) -> bool:
        """Write ``payload`` (JSON-serializable) to the cache. Best-effort."""
        path = self._path(key)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)
            return True
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("GIS cache write failed for %s: %s", key, exc)
            return False

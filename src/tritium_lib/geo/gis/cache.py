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

#: Only these suffixes are ever deleted by :meth:`GISCache.sweep` — a hard
#: guard so an unexpected file (a ``.txt`` note, a lock file) is never removed.
_SWEEP_SUFFIXES = {".json", ".bin"}


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

    # -- retention ----------------------------------------------------------
    def sweep(
        self,
        *,
        retention_days: float,
        max_total_bytes: int,
        now: float | None = None,
    ) -> dict:
        """Bound the on-disk cache by age then by total size, oldest-first.

        Mirrors ``tritium_lib.recording.retention.sweep_recordings`` (on ``dev``,
        commit ``ffbfd6a``): two passes, both ordered oldest-first by mtime,
        caller-supplied bounds, safe unlink.  Generalized for the GIS cache — it
        walks ``cache_dir`` **recursively** (so the ``tiles/`` XYZ tree is
        covered) and only ever deletes files whose suffix is in
        :data:`_SWEEP_SUFFIXES` and which resolve to *inside* ``cache_dir`` (an
        out-of-tree symlink is never followed for deletion).  A later
        unification of the two sweeps should be mechanical.

        Passes:

        1. **Age** — delete files older than ``retention_days`` (skipped when
           ``retention_days <= 0``).
        2. **Size** — while the survivors still exceed ``max_total_bytes``,
           delete oldest-first until under the cap (skipped when
           ``max_total_bytes <= 0``).

        Empty subdirectories left behind are pruned best-effort.  A missing
        cache dir is a safe no-op.  Never raises — returns
        ``{"removed": [relpaths], "freed_bytes": int, "remaining_bytes": int}``.

        Since 2026-07-10 (tick 2) this delegates to the shared
        :func:`tritium_lib.store.retention.sweep_dir` core — the
        "later unification" promised above, done.
        """
        from tritium_lib.store.retention import sweep_dir

        return sweep_dir(
            self.cache_dir,
            retention_days=retention_days,
            max_total_bytes=max_total_bytes,
            suffixes=_SWEEP_SUFFIXES,
            recursive=True,
            prune_empty=True,
            now=now,
            label="GIS cache",
        )

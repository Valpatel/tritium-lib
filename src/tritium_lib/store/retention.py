# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared disk-retention sweep for on-disk file stores.

One implementation of the age+size bound used by every directory-shaped
store in the system:

- ``tritium_lib.recording.retention.sweep_recordings`` — flat AAR/JSONL
  recording stores (``data/sim_recordings/``)
- ``tritium_lib.geo.gis.cache.GISCache.sweep`` — the GIS layer cache,
  including its recursive ``tiles/`` XYZ tree

Both were near-identical twins (the GIS one said "a later unification
of the two sweeps should be mechanical" — this is that unification,
2026-07-10).  Policy:

1. **Age pass** — delete files older than ``retention_days`` (skipped
   when ``retention_days <= 0``), oldest-first by mtime.
2. **Size pass** — while the survivors still exceed
   ``max_total_bytes``, delete oldest-first until under the cap
   (skipped when ``max_total_bytes <= 0``).

Safety:

- only files whose suffix is in the caller's ``suffixes`` allowlist are
  ever deleted;
- a candidate must *resolve* to inside the swept directory — an
  out-of-tree symlink target is never deleted;
- everything is best-effort: a broken store never raises, the sweep
  just skips what it cannot stat/delete.

Returns ``{"removed": [names], "freed_bytes": int, "remaining_bytes": int}``
where names are relative paths for recursive sweeps and bare filenames
for flat ones.  A missing directory is a safe no-op.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

__all__ = ["sweep_dir"]


def _safe_unlink(path: Path, base_resolved: Path, suffixes: frozenset[str]) -> bool:
    """Delete *path* only if it resolves inside *base_resolved* with an
    allowlisted suffix.  Never follows an out-of-tree symlink target."""
    try:
        resolved = path.resolve()
        resolved.relative_to(base_resolved)  # ValueError if outside the tree
        if resolved.suffix.lower() not in suffixes:
            return False
        resolved.unlink()
        return True
    except (OSError, ValueError) as exc:
        logger.debug("retention sweep: unlink skipped for %s: %s", path, exc)
        return False


def _prune_empty_dirs(base: Path) -> None:
    """Remove now-empty subdirectories of *base* (deepest-first). Best-effort."""
    try:
        subdirs = [p for p in base.rglob("*") if p.is_dir()]
    except OSError:
        return
    for d in sorted(subdirs, key=lambda p: len(p.parts), reverse=True):
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except OSError:
            continue


def sweep_dir(
    directory: Path | str,
    *,
    retention_days: float,
    max_total_bytes: int,
    suffixes: Iterable[str],
    recursive: bool = False,
    prune_empty: bool = False,
    now: float | None = None,
    label: str = "retention",
) -> dict[str, Any]:
    """Bound the file store under *directory* by age then total size.

    Args:
        directory: The store root.  Missing dir is a safe no-op.
        retention_days: Age bound in days; ``<= 0`` skips the age pass.
        max_total_bytes: Total-size cap in bytes; ``<= 0`` skips the
            size pass.
        suffixes: Allowlist of file suffixes (with dot, lowercase) that
            may be deleted — anything else is never touched.
        recursive: Walk the whole tree (``rglob``) instead of only the
            direct children (``glob``).
        prune_empty: After deleting, remove now-empty subdirectories
            (only meaningful with ``recursive=True``).
        now: Override the current time (testing).
        label: Log prefix so call-sites stay distinguishable.
    """
    base = Path(directory)
    eff_now = time.time() if now is None else float(now)
    suffix_set = frozenset(s.lower() for s in suffixes)
    empty = {"removed": [], "freed_bytes": 0, "remaining_bytes": 0}

    try:
        if not base.is_dir():
            return empty
        base_resolved = base.resolve()
    except OSError:
        return empty

    entries: list[tuple[Path, str, float, int]] = []
    try:
        walker = base.rglob("*") if recursive else base.glob("*")
        for path in walker:
            try:
                if not path.is_file():
                    continue
                if path.suffix.lower() not in suffix_set:
                    continue
                st = path.stat()
                relpath = str(path.relative_to(base)) if recursive else path.name
            except (OSError, ValueError):
                continue
            entries.append((path, relpath, st.st_mtime, st.st_size))
    except OSError as exc:
        logger.debug("%s sweep walk failed: %s", label, exc)
        return empty

    removed: list[str] = []
    freed = 0

    # Pass 1: age-based eviction.
    if retention_days > 0:
        cutoff = eff_now - retention_days * 86400.0
        survivors: list[tuple[Path, str, float, int]] = []
        for path, relpath, mtime, size in entries:
            if mtime < cutoff and _safe_unlink(path, base_resolved, suffix_set):
                removed.append(relpath)
                freed += size
            else:
                survivors.append((path, relpath, mtime, size))
        entries = survivors

    # Pass 2: size-based eviction, oldest-first.
    total = sum(size for _, _, _, size in entries)
    if max_total_bytes > 0 and total > max_total_bytes:
        for path, relpath, _mtime, size in sorted(entries, key=lambda e: e[2]):
            if total <= max_total_bytes:
                break
            if _safe_unlink(path, base_resolved, suffix_set):
                removed.append(relpath)
                freed += size
                total -= size

    if prune_empty:
        _prune_empty_dirs(base)

    if removed:
        logger.info(
            "%s sweep: removed %d file(s), freed %.1f MB (remaining %.1f MB)",
            label, len(removed), freed / 1e6, total / 1e6,
        )
    return {"removed": removed, "freed_bytes": freed, "remaining_bytes": total}

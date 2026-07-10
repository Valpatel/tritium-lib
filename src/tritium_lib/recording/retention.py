# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Disk retention for JSONL recording stores.

Bounds an on-disk directory of ``*.jsonl`` recording files by age and/or
total size, deleting oldest-first by mtime.  Extracted from tritium-sc
(``app/routers/sim_recordings.py``) so the sweep can run wherever
recordings are *written* — the AAR recorder, the sim-recordings API, a
long-running server's periodic sweep thread — not only inside a live
server.  Recordings written by short-lived processes (tests, headless
battle runs) previously never got swept; the 2026-07-10 audit found a
3.1 GB store whose files were 19-20 days old under a 7-day policy
because the only sweep call-site was a server daemon thread that wasn't
running.

Pure function: no environment reads, no global directory default — the
caller supplies the directory and both bounds.  Deletion is
hard-restricted to ``.jsonl`` files directly inside the target
directory (see :func:`_safe_unlink`).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["sweep_recordings"]


def _safe_unlink(path: Path, base: Path) -> bool:
    """Delete *path* only if it is a ``.jsonl`` file directly inside *base*.

    Hard-restricts deletion to the recordings directory: the resolved
    parent must equal the resolved base and the suffix must be
    ``.jsonl``.  Never follows paths outside the recordings dir.
    """
    try:
        resolved = path.resolve()
        if resolved.parent != base.resolve():
            return False
        if resolved.suffix != ".jsonl":
            return False
        resolved.unlink()
        return True
    except OSError as exc:
        logger.debug("sweep_recordings: unlink failed for %s: %s", path, exc)
        return False


def sweep_recordings(
    directory: Path | str,
    *,
    retention_days: float,
    max_total_bytes: int,
    now: float | None = None,
) -> dict[str, Any]:
    """Bound the on-disk JSONL recording store under *directory*.

    Two passes, both oldest-first by mtime, restricted to ``*.jsonl``
    files directly inside *directory*:

    1. **Age** — delete files older than ``retention_days`` (skipped if
       the value is <= 0).
    2. **Size** — if the surviving total still exceeds
       ``max_total_bytes``, delete oldest-first until under the cap
       (skipped if <= 0).

    Returns ``{"removed": [names], "freed_bytes": int, "remaining_bytes": int}``.
    A missing directory is a safe no-op.
    """
    base = Path(directory)
    eff_now = time.time() if now is None else float(now)

    if not base.is_dir():
        return {"removed": [], "freed_bytes": 0, "remaining_bytes": 0}

    entries: list[tuple[Path, float, int]] = []
    for path in base.glob("*.jsonl"):
        if not path.is_file():
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        entries.append((path, st.st_mtime, st.st_size))

    removed: list[str] = []
    freed = 0

    # Pass 1: age-based eviction.
    if retention_days > 0:
        cutoff = eff_now - retention_days * 86400.0
        survivors: list[tuple[Path, float, int]] = []
        for path, mtime, size in entries:
            if mtime < cutoff and _safe_unlink(path, base):
                removed.append(path.name)
                freed += size
            else:
                survivors.append((path, mtime, size))
        entries = survivors

    # Pass 2: size-based eviction, oldest-first.
    total = sum(size for _, _, size in entries)
    if max_total_bytes > 0 and total > max_total_bytes:
        for path, _mtime, size in sorted(entries, key=lambda e: e[1]):
            if total <= max_total_bytes:
                break
            if _safe_unlink(path, base):
                removed.append(path.name)
                freed += size
                total -= size

    if removed:
        logger.info(
            "sim_recordings sweep: removed %d file(s), freed %.1f MB "
            "(remaining %.1f MB)",
            len(removed), freed / 1e6, total / 1e6,
        )
    return {"removed": removed, "freed_bytes": freed, "remaining_bytes": total}

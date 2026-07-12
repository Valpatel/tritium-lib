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

Since 2026-07-10 (tick 2) this is a thin wrapper over the shared
:func:`tritium_lib.store.retention.sweep_dir` core, which also backs
``GISCache.sweep`` — one retention implementation for every
directory-shaped store.  Pure function: no environment reads, the
caller supplies the directory and both bounds.  Deletion is
hard-restricted to ``.jsonl`` files directly inside the target
directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from tritium_lib.store.retention import sweep_dir

__all__ = ["sweep_recordings"]


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
    return sweep_dir(
        directory,
        retention_days=retention_days,
        max_total_bytes=max_total_bytes,
        suffixes=(".jsonl",),
        recursive=False,
        prune_empty=False,
        now=now,
        label="sim_recordings",
    )

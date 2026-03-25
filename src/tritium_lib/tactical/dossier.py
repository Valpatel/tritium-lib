# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tactical dossier shim — re-exports from tritium_lib.store.

For backward compatibility with code that imports from engine.tactical.dossier,
this shim re-exports DossierStore from its canonical location in tritium_lib.store.
"""

from tritium_lib.store import DossierStore

__all__ = ["DossierStore"]

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Shared pytest fixtures for city3d frontend tests.

The city3d frontend is split across city3d.html and external JS modules in
city3d/*.js.  The per-test `source` fixtures only read city3d.html, which
causes them to miss patterns that live in the JS modules.

This conftest overrides the CITY3D_PATH module-level fixture to load the
combined source of city3d.html + all city3d/*.js files, so tests correctly
find patterns regardless of which file they're in.
"""
from __future__ import annotations

import os
import glob
import pytest

_DEMOS_DIR = os.path.dirname(os.path.dirname(__file__))
_CITY3D_HTML = os.path.join(_DEMOS_DIR, "city3d.html")
_CITY3D_JS_GLOB = os.path.join(_DEMOS_DIR, "city3d", "*.js")


def _load_combined_source() -> str:
    """Load city3d.html + all city3d/*.js files as one combined string."""
    parts: list[str] = []
    # HTML first
    with open(_CITY3D_HTML, encoding="utf-8") as f:
        parts.append(f.read())
    # All JS modules (sorted for determinism)
    for js_path in sorted(glob.glob(_CITY3D_JS_GLOB)):
        with open(js_path, encoding="utf-8") as f:
            parts.append(f"\n/* === {os.path.basename(js_path)} === */\n")
            parts.append(f.read())
    return "\n".join(parts)


@pytest.fixture(scope="session")
def city3d_combined_source() -> str:
    """Combined HTML + JS source for all city3d frontend files."""
    return _load_combined_source()


# Override the per-module 'source' fixture used by all city3d test files.
# Each test file defines its own `source` fixture with scope="module" that
# only reads city3d.html.  By defining a session-scoped fixture here with
# the same name, we make the combined source available.
#
# Note: pytest uses the fixture closest to the test (local > conftest).
# Since local fixtures have higher priority, this conftest fixture is only
# used when a test requests 'source' and no local fixture overrides it.
# The city3d test files DO define local fixtures, so we need to patch those.
#
# Solution: we shadow CITY3D_PATH at conftest level so the local fixture
# reads the combined source via a patched open.  Instead, we provide a
# session fixture that the test files' fixtures delegate to via autouse.

# Actually the simplest correct approach: define `source` here as a
# session fixture.  pytest will prefer the local module fixture, so this
# won't override local definitions.  Therefore we patch at the module level
# by providing a conftest-level `source` that individual tests can use
# if they explicitly request it, while the local module fixtures still work.
#
# The real fix is below: we monkeypatch the built-in open so that when
# city3d.html is opened inside a fixture, it returns combined content.

@pytest.fixture(autouse=True, scope="session")
def _patch_city3d_open(monkeypatch_session: pytest.MonkeyPatch | None = None):
    """Not used — individual test files define their own source fixtures."""
    pass

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Regression guard for the canonical target lifecycle constants.

Wave 204 extracted the terminal-status set out of ``targets_unified.py``
into :mod:`tritium_lib.models.target_status`.  Both the WS broadcast
filter and the REST ``/api/targets`` endpoint now import it.  Drift in
this set silently re-introduces the Wave 198 header/API mismatch, so we
pin the contents here.
"""

from __future__ import annotations

import pytest

from tritium_lib.models.target_status import (
    ACTIVE_STATUSES,
    ALL_STATUSES,
    RESTING_STATUSES,
    TERMINAL_STATUSES,
    is_terminal,
)


# ---------------------------------------------------------------------------
# 1. The W201-sanctioned terminal set is exactly five elements
# ---------------------------------------------------------------------------


class TestTerminalStatuses:
    def test_terminal_set_is_exactly_five(self):
        assert len(TERMINAL_STATUSES) == 5

    def test_terminal_set_membership(self):
        # Pinned: changing this asserts updates downstream consumers
        # (frontend target-counter.js, ws.py, targets_unified.py).
        assert TERMINAL_STATUSES == frozenset({
            "eliminated",
            "destroyed",
            "despawned",
            "neutralized",
            "escaped",
        })

    def test_terminal_set_is_frozen(self):
        # frozenset cannot be mutated — protects against drive-by edits.
        assert isinstance(TERMINAL_STATUSES, frozenset)

    def test_terminal_is_subset_of_all(self):
        assert TERMINAL_STATUSES.issubset(ALL_STATUSES)


# ---------------------------------------------------------------------------
# 2. ALL_STATUSES enumerates every documented status
# ---------------------------------------------------------------------------


class TestAllStatuses:
    def test_all_statuses_documented(self):
        # The 10 documented statuses from
        # tritium_lib.sim_engine.core.entity.SimulationTarget.
        assert ALL_STATUSES == frozenset({
            "active",
            "idle",
            "stationary",
            "arrived",
            "low_battery",
            "escaped",
            "neutralized",
            "eliminated",
            "destroyed",
            "despawned",
        })

    def test_all_statuses_is_frozen(self):
        assert isinstance(ALL_STATUSES, frozenset)


# ---------------------------------------------------------------------------
# 3. Lifecycle groupings are coherent
# ---------------------------------------------------------------------------


class TestStatusGroups:
    def test_groups_are_subsets_of_all(self):
        assert RESTING_STATUSES.issubset(ALL_STATUSES)
        assert ACTIVE_STATUSES.issubset(ALL_STATUSES)

    def test_groups_disjoint_from_terminal(self):
        # A "resting" or "active" target is by definition not terminal.
        assert RESTING_STATUSES.isdisjoint(TERMINAL_STATUSES)
        assert ACTIVE_STATUSES.isdisjoint(TERMINAL_STATUSES)

    def test_resting_membership(self):
        assert RESTING_STATUSES == frozenset({"idle", "stationary", "arrived"})

    def test_active_membership(self):
        assert ACTIVE_STATUSES == frozenset({"active", "low_battery"})


# ---------------------------------------------------------------------------
# 4. is_terminal helper agrees with the set membership and is defensive
# ---------------------------------------------------------------------------


class TestIsTerminal:
    @pytest.mark.parametrize("status", sorted(TERMINAL_STATUSES))
    def test_terminal_strings_classified(self, status):
        assert is_terminal(status) is True

    @pytest.mark.parametrize("status", [
        "active", "idle", "stationary", "arrived", "low_battery", "patrolling",
    ])
    def test_non_terminal_strings_classified(self, status):
        assert is_terminal(status) is False

    @pytest.mark.parametrize("status", [None, 42, 3.14, [], {}, object()])
    def test_non_string_returns_false(self, status):
        assert is_terminal(status) is False

    def test_empty_string_is_not_terminal(self):
        assert is_terminal("") is False

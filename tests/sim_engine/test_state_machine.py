# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.sim_engine.core.state_machine."""

import pytest

from tritium_lib.sim_engine.core.state_machine import State, StateMachine, Transition


class TestStateMachineBuilder:
    def test_basic_creation(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        assert sm.current_state == "idle"

    def test_initial_state_required(self):
        with pytest.raises(ValueError, match="initial_state"):
            StateMachine(None)

    def test_transition_on_condition(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("alert"))
        sm.add_transition("idle", "alert", lambda ctx: ctx.get("enemy", False))
        sm.tick(0.1, {"enemy": True})
        assert sm.current_state == "alert"

    def test_no_transition_when_condition_false(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("alert"))
        sm.add_transition("idle", "alert", lambda ctx: ctx.get("enemy", False))
        sm.tick(0.1, {"enemy": False})
        assert sm.current_state == "idle"

    def test_on_enter_called(self):
        entered = []
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("alert", on_enter=lambda ctx: entered.append("alert")))
        sm.add_transition("idle", "alert", lambda ctx: True)
        sm.tick(0.1, {})
        assert "alert" in entered

    def test_on_exit_called(self):
        exited = []
        sm = StateMachine("idle")
        sm.add_state(State("idle", on_exit=lambda ctx: exited.append("idle")))
        sm.add_state(State("alert"))
        sm.add_transition("idle", "alert", lambda ctx: True)
        sm.tick(0.1, {})
        assert "idle" in exited

    def test_on_tick_called(self):
        ticked = []
        sm = StateMachine("idle")
        sm.add_state(State("idle", on_tick=lambda ctx, dt: ticked.append(dt)))
        sm.tick(0.5, {})
        assert len(ticked) == 1
        assert ticked[0] == 0.5

    def test_time_in_state(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.tick(0.1, {})
        sm.tick(0.2, {})
        assert abs(sm.time_in_state - 0.3) < 0.01

    def test_time_resets_on_transition(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("alert"))
        sm.add_transition("idle", "alert", lambda ctx: ctx.get("go", False))
        sm.tick(0.5, {})
        sm.tick(0.1, {"go": True})
        # After transition, time_in_state should be reset
        assert sm.time_in_state < 0.2

    def test_history(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("alert"))
        sm.add_transition("idle", "alert", lambda ctx: True)
        sm.tick(0.1, {})
        history = sm.history
        assert len(history) == 1
        assert history[0][1] == "idle"
        assert history[0][2] == "alert"

    def test_force_state(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("alert"))
        sm.tick(0.1, {})  # Initial on_enter
        sm.force_state("alert")
        assert sm.current_state == "alert"

    def test_force_state_unknown_raises(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.tick(0.1, {})
        with pytest.raises(ValueError, match="not found"):
            sm.force_state("nonexistent")

    def test_min_duration_blocks_transitions(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle", min_duration=1.0))
        sm.add_state(State("alert"))
        sm.add_transition("idle", "alert", lambda ctx: True)
        sm.tick(0.1, {})  # 0.1s < 1.0s min_duration
        assert sm.current_state == "idle"
        # After enough time, transition should fire
        sm.tick(1.0, {})
        assert sm.current_state == "alert"

    def test_max_duration_auto_transition(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle", max_duration=0.5, max_duration_target="alert"))
        sm.add_state(State("alert"))
        sm.tick(0.1, {})  # Initial on_enter
        sm.tick(0.5, {})  # Exceeds max_duration
        assert sm.current_state == "alert"

    def test_guard_blocks_transition(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("alert"))
        sm.add_transition(
            "idle", "alert",
            condition=lambda ctx: True,
            guard=lambda ctx: ctx.get("allowed", False),
        )
        sm.tick(0.1, {"allowed": False})
        assert sm.current_state == "idle"
        sm.tick(0.1, {"allowed": True})
        assert sm.current_state == "alert"

    def test_priority_transitions(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("low"))
        sm.add_state(State("high"))
        sm.add_transition("idle", "low", lambda ctx: True, priority=1)
        sm.add_transition("idle", "high", lambda ctx: True, priority=10)
        sm.tick(0.1, {})
        assert sm.current_state == "high"

    def test_state_names(self):
        sm = StateMachine("idle")
        sm.add_state(State("idle"))
        sm.add_state(State("alert"))
        sm.add_state(State("combat"))
        assert set(sm.state_names) == {"idle", "alert", "combat"}


class TestStateMachineLegacy:
    def test_legacy_constructor(self):
        states = [State("a"), State("b")]
        transitions = [Transition("a", "b", lambda: True)]
        sm = StateMachine(
            states=states,
            transitions=transitions,
            initial_state="a",
        )
        sm.tick(0.1)
        assert sm.current_state == "b"

    def test_legacy_no_transition_when_false(self):
        states = [State("a"), State("b")]
        transitions = [Transition("a", "b", lambda: False)]
        sm = StateMachine(
            states=states,
            transitions=transitions,
            initial_state="a",
        )
        sm.tick(0.1)
        assert sm.current_state == "a"

    def test_legacy_invalid_initial_state(self):
        states = [State("a")]
        with pytest.raises(ValueError, match="not found"):
            StateMachine(states=states, transitions=[], initial_state="xyz")

    def test_legacy_on_transition_callback(self):
        called = []
        states = [State("a"), State("b")]
        transitions = [Transition("a", "b", lambda: True,
                                  on_transition=lambda: called.append(True))]
        sm = StateMachine(states=states, transitions=transitions, initial_state="a")
        sm.tick(0.1)
        assert called == [True]

    def test_legacy_tick_without_context(self):
        """Legacy mode tick works without a context dict."""
        states = [State("a")]
        sm = StateMachine(states=states, transitions=[], initial_state="a")
        sm.tick(0.1)  # Should not raise


class TestState:
    def test_state_name(self):
        s = State("idle")
        assert s.name == "idle"

    def test_on_enter_callback_no_args(self):
        """on_enter with no-arg callback (legacy compat)."""
        called = []
        s = State("test", on_enter=lambda: called.append(True))
        s.on_enter({})
        assert called == [True]

    def test_on_enter_callback_with_ctx(self):
        """on_enter with ctx argument."""
        received = []
        s = State("test", on_enter=lambda ctx: received.append(ctx))
        s.on_enter({"key": "val"})
        assert received[0] == {"key": "val"}

    def test_tick_returns_none_by_default(self):
        s = State("idle")
        result = s.tick(0.1, {})
        assert result is None

    def test_min_max_duration_attributes(self):
        s = State("test", min_duration=1.0, max_duration=5.0, max_duration_target="next")
        assert s.min_duration == 1.0
        assert s.max_duration == 5.0
        assert s.max_duration_target == "next"

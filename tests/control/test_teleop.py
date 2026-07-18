# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the gamepad -> twist seam.

The properties worth pinning are the ones that bite in the field: the deadzone
must not step at its own edge, an absent enable button must refuse to drive
rather than grant permission, a stale input must decay to a stop on its own,
and the ControlIntent/TwistCommand yaw-sign mismatch must be negated exactly
once.

Axes here are in the *neutral* convention the ROS2 teleop already uses
(stick up = +forward, stick right = +turn/clockwise); per-driver adapters
normalize into it upstream.
"""

import pytest

from tritium_lib.control import (
    AxisMap,
    GamepadState,
    SlewLimiter,
    TeleopProfile,
    TeleopWatchdog,
    TwistCommand,
    apply_deadzone,
    apply_expo,
    shape_axis,
    twist_command_from_intent,
    twist_from_stick,
)

PROFILE = TeleopProfile(max_linear_mps=1.0, max_angular_rps=2.0, deadzone=0.10)


def _pad(linear=0.0, angular=0.0, buttons=(), t=0.0):
    """Neutral-convention frame: axis 0 = yaw, axis 1 = forward."""
    return GamepadState(axes=(angular, linear), buttons=tuple(buttons), timestamp_s=t)


# --------------------------------------------------------------------------
# axis shaping
# --------------------------------------------------------------------------


def test_deadzone_zeroes_a_resting_stick():
    assert apply_deadzone(0.05, 0.10) == 0.0
    assert apply_deadzone(-0.05, 0.10) == 0.0


def test_deadzone_rescales_so_there_is_no_step_at_the_edge():
    """The detail a naive implementation gets wrong.

    Subtracting the deadzone without rescaling leaves output jumping from 0 to
    nothing and never reaching 1.0. Rescaling makes it continuous at the
    boundary and still hit full authority at full deflection.
    """
    assert apply_deadzone(0.1001, 0.10) == pytest.approx(0.0, abs=1e-3)
    assert apply_deadzone(1.0, 0.10) == pytest.approx(1.0)
    assert apply_deadzone(-1.0, 0.10) == pytest.approx(-1.0)
    # Halfway between the deadzone edge and full travel -> halfway out.
    assert apply_deadzone(0.55, 0.10) == pytest.approx(0.5)


def test_deadzone_is_monotonic():
    prev = -1.1
    for i in range(101):
        value = apply_deadzone(i / 100.0, 0.10)
        assert value >= prev
        prev = value


def test_axis_beyond_unit_range_is_clamped():
    """Some drivers report slightly outside [-1, 1]; that must not exceed max."""
    assert apply_deadzone(1.4, 0.10) == pytest.approx(1.0)
    assert apply_deadzone(-1.4, 0.10) == pytest.approx(-1.0)


def test_expo_zero_is_identity():
    for value in (-1.0, -0.3, 0.0, 0.42, 1.0):
        assert apply_expo(value, 0.0) == pytest.approx(value)


def test_expo_preserves_endpoints_and_softens_the_middle():
    assert apply_expo(0.0, 0.6) == pytest.approx(0.0)
    assert apply_expo(1.0, 0.6) == pytest.approx(1.0)
    assert apply_expo(-1.0, 0.6) == pytest.approx(-1.0)
    assert abs(apply_expo(0.5, 0.6)) < 0.5


def test_expo_is_odd_and_monotonic():
    prev = -1.1
    for i in range(-100, 101):
        value = i / 100.0
        out = apply_expo(value, 0.75)
        assert out == pytest.approx(-apply_expo(-value, 0.75))
        assert out >= prev
        prev = out


def test_shape_axis_is_deadzone_then_expo():
    assert shape_axis(0.55, 0.10, 0.75) == pytest.approx(
        apply_expo(apply_deadzone(0.55, 0.10), 0.75))


# --------------------------------------------------------------------------
# stick -> twist
# --------------------------------------------------------------------------


def test_centered_sticks_command_a_stop():
    assert twist_from_stick(_pad(), PROFILE) == TwistCommand.stop()


def test_full_forward_gives_max_linear_and_no_yaw():
    twist = twist_from_stick(_pad(linear=1.0), PROFILE)
    assert twist.linear_mps == pytest.approx(1.0)
    assert twist.angular_rps == pytest.approx(0.0)


def test_pulling_the_stick_back_commands_reverse():
    assert twist_from_stick(_pad(linear=-1.0), PROFILE).linear_mps == pytest.approx(-1.0)


def test_stick_right_turns_to_starboard_per_rep103():
    """Neutral stick-right is clockwise; TwistCommand counts port positive."""
    assert twist_from_stick(_pad(angular=1.0), PROFILE).angular_rps == pytest.approx(-2.0)
    assert twist_from_stick(_pad(angular=-1.0), PROFILE).angular_rps == pytest.approx(2.0)


def test_axis_inversion_is_configurable_data_not_a_converter_per_pad():
    profile = TeleopProfile(
        max_linear_mps=1.0, max_angular_rps=2.0,
        axes=AxisMap(linear_axis=1, angular_axis=0, linear_inverted=True),
    )
    assert twist_from_stick(_pad(linear=1.0), profile).linear_mps == pytest.approx(-1.0)


def test_missing_axis_reads_as_centered_not_an_error():
    """A pad with fewer axes than configured must degrade, not crash."""
    state = GamepadState(axes=(0.0,), buttons=(), timestamp_s=0.0)
    assert twist_from_stick(state, PROFILE) == TwistCommand.stop()


# --------------------------------------------------------------------------
# enable (deadman) + turbo
# --------------------------------------------------------------------------


def test_enable_button_dominates_the_sticks():
    profile = TeleopProfile(max_linear_mps=1.0, max_angular_rps=2.0, enable_button=4)
    assert twist_from_stick(
        _pad(linear=1.0, buttons=(False,) * 8), profile) == TwistCommand.stop()

    buttons = [False] * 8
    buttons[4] = True
    assert twist_from_stick(
        _pad(linear=1.0, buttons=buttons), profile).linear_mps == pytest.approx(1.0)


def test_absent_enable_button_refuses_to_drive():
    """Fail-safe direction: a short pad must not be read as permission."""
    profile = TeleopProfile(max_linear_mps=1.0, max_angular_rps=2.0, enable_button=4)
    assert twist_from_stick(_pad(linear=1.0, buttons=()), profile) == TwistCommand.stop()


def test_turbo_raises_the_ceiling_while_held():
    profile = TeleopProfile(
        max_linear_mps=1.0, max_angular_rps=2.0,
        turbo_linear_mps=2.5, turbo_angular_rps=3.0, turbo_button=5,
    )
    assert twist_from_stick(_pad(linear=1.0), profile).linear_mps == pytest.approx(1.0)
    buttons = [False] * 8
    buttons[5] = True
    fast = twist_from_stick(_pad(linear=1.0, angular=-1.0, buttons=buttons), profile)
    assert fast.linear_mps == pytest.approx(2.5)
    assert fast.angular_rps == pytest.approx(3.0)


def test_turbo_without_a_configured_ceiling_is_a_no_op():
    profile = TeleopProfile(max_linear_mps=1.0, max_angular_rps=2.0, turbo_button=5)
    buttons = [False] * 8
    buttons[5] = True
    assert twist_from_stick(
        _pad(linear=1.0, buttons=buttons), profile).linear_mps == pytest.approx(1.0)


# --------------------------------------------------------------------------
# the yaw-sign bridge
# --------------------------------------------------------------------------


def test_intent_to_twist_negates_the_yaw_sign_exactly_once():
    """ControlIntent.turn is +clockwise; TwistCommand.angular_rps is +port."""
    from tritium_lib.models.body import ControlIntent

    twist = twist_command_from_intent(
        ControlIntent(forward=1.0, turn=1.0), max_linear_mps=2.0, max_angular_rps=3.0)
    assert twist.linear_mps == pytest.approx(2.0)
    assert twist.angular_rps == pytest.approx(-3.0)


def test_intent_bridge_agrees_with_the_stick_path_on_direction():
    """Both routes into a twist must turn the body the same way."""
    from tritium_lib.models.body import ControlIntent

    from_stick = twist_from_stick(_pad(angular=1.0), PROFILE)
    from_intent = twist_command_from_intent(
        ControlIntent(forward=0.0, turn=1.0), max_linear_mps=1.0, max_angular_rps=2.0)
    assert from_stick.angular_rps == pytest.approx(from_intent.angular_rps)


# --------------------------------------------------------------------------
# watchdog — the production safety property
# --------------------------------------------------------------------------


def test_stale_input_decays_to_a_stop():
    """A dropped pad or dead link must not latch the last command."""
    dog = TeleopWatchdog(timeout_s=0.5)
    dog.feed(TwistCommand(1.0, 0.0), now_s=10.0)
    assert dog.poll(now_s=10.4).linear_mps == pytest.approx(1.0)
    assert dog.poll(now_s=10.6) == TwistCommand.stop()


def test_expiry_stays_tripped_until_a_fresh_frame_arrives():
    """Un-tripping on the clock alone would let a dead link stutter the body."""
    dog = TeleopWatchdog(timeout_s=0.5)
    dog.feed(TwistCommand(1.0, 0.0), now_s=0.0)
    assert dog.poll(now_s=1.0) == TwistCommand.stop()
    assert dog.poll(now_s=1.01) == TwistCommand.stop()
    assert dog.feed(TwistCommand(1.0, 0.0), now_s=1.02).linear_mps == pytest.approx(1.0)
    assert dog.poll(now_s=1.03).linear_mps == pytest.approx(1.0)


def test_poll_before_any_input_is_a_stop():
    dog = TeleopWatchdog(timeout_s=0.5)
    assert dog.poll(now_s=0.0) == TwistCommand.stop()
    assert dog.expired


def test_watchdog_can_be_disabled_explicitly():
    dog = TeleopWatchdog(timeout_s=None)
    dog.feed(TwistCommand(1.0, 0.0), now_s=0.0)
    assert dog.poll(now_s=1e6).linear_mps == pytest.approx(1.0)


def test_non_positive_timeout_is_rejected():
    with pytest.raises(ValueError):
        TeleopWatchdog(timeout_s=0.0)


# --------------------------------------------------------------------------
# slew limiting
# --------------------------------------------------------------------------


def test_slew_limiter_caps_acceleration():
    """A stick slammed to full must ramp, not step."""
    lim = SlewLimiter(max_linear_accel=2.0, max_angular_accel=4.0)
    out = lim.limit(TwistCommand(1.0, 2.0), dt_s=0.1)
    assert out.linear_mps == pytest.approx(0.2)
    assert out.angular_rps == pytest.approx(0.4)
    assert lim.limit(TwistCommand(1.0, 2.0), dt_s=0.1).linear_mps == pytest.approx(0.4)


def test_slew_limiter_reaches_the_target_and_holds_it():
    lim = SlewLimiter(max_linear_accel=2.0, max_angular_accel=4.0)
    for _ in range(100):
        out = lim.limit(TwistCommand(0.5, 0.0), dt_s=0.1)
    assert out.linear_mps == pytest.approx(0.5)


def test_slew_limiter_bounds_deceleration_symmetrically():
    lim = SlewLimiter(max_linear_accel=2.0, max_angular_accel=4.0)
    for _ in range(50):
        lim.limit(TwistCommand(1.0, 0.0), dt_s=0.1)
    assert lim.limit(TwistCommand.stop(), dt_s=0.1).linear_mps == pytest.approx(0.8)


def test_zero_dt_cannot_bypass_the_limiter():
    lim = SlewLimiter(max_linear_accel=2.0, max_angular_accel=4.0)
    assert lim.limit(TwistCommand(1.0, 2.0), dt_s=0.0) == TwistCommand.stop()


def test_emergency_stop_ignores_the_ramp():
    """A limiter that also smooths the e-stop is a bug, not a feature."""
    lim = SlewLimiter(max_linear_accel=2.0, max_angular_accel=4.0)
    for _ in range(50):
        lim.limit(TwistCommand(1.0, 0.0), dt_s=0.1)
    assert lim.current.linear_mps == pytest.approx(1.0)
    assert lim.emergency_stop() == TwistCommand.stop()
    # and it resumes ramping from zero, not from the pre-stop speed
    assert lim.limit(TwistCommand(1.0, 0.0), dt_s=0.1).linear_mps == pytest.approx(0.2)


def test_slew_limiter_rejects_non_positive_limits():
    with pytest.raises(ValueError):
        SlewLimiter(max_linear_accel=0.0, max_angular_accel=1.0)


# --------------------------------------------------------------------------
# end to end — stick all the way to the body mixer
# --------------------------------------------------------------------------


def test_teleop_output_feeds_the_existing_differential_mixer():
    """Teleop must land on the same mixer the route follower already uses."""
    from tritium_lib.control import differential_stride

    twist = twist_from_stick(_pad(linear=1.0, angular=-1.0), PROFILE)
    bias = differential_stride(twist, track_width_m=0.26, nominal_mps=1.0)
    # Turning to port: the port (left) side travels slower than starboard.
    assert bias.left_scale < bias.right_scale

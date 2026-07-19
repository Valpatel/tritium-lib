# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the closed-loop gait speed tracker.

All time is INJECTED — no wall-clock reads anywhere (the determinism suite
forbids them).  Every "plant" here is a scripted function of the demand, so
these tests prove the controller's ARITHMETIC: inertness on an achievable
setpoint, convergence to the stop without windup on an unachievable one,
authority limits that hold under adversarial input, and the honest-ceiling
report.  They are NOT evidence the controller helps a real body — the
module docstring keeps that ledger, and the StepReflex history is why the
distinction is load-bearing.
"""

from __future__ import annotations

import math

import pytest

from tritium_lib.control import (
    GaitPhaseClock,
    GaitSpeedState,
    GaitSpeedTracker,
    StrideSpeedEstimator,
)
from tritium_lib.control.gait_speed import (
    DEFAULT_MAX_CADENCE_SCALE,
    DEFAULT_MIN_AMP_SCALE,
    DEFAULT_MIN_CADENCE_SCALE,
)

NOMINAL = 1.6  # default trot table speed the trajectory generator ratios by
DT = 0.4       # ~one trot stride at 2.6 Hz — the recommended update period


def run_plant(
    tracker: GaitSpeedTracker,
    commanded: float,
    plant,
    ticks: int,
    dt: float = DT,
    state: GaitSpeedState | None = None,
    demand0: float | None = None,
):
    """Drive tracker against ``plant(demand) -> measured``, return commands.

    The measurement each tick is the plant's response to the PREVIOUS
    tick's demand — the same one-tick attribution the slip guard assumes.
    """
    cmds = []
    demand = commanded if demand0 is None else demand0
    for _ in range(ticks):
        measured = plant(demand)
        cmd = tracker.track(commanded, measured, dt, state=state)
        state = cmd.state
        demand = cmd.demand_mps
        cmds.append(cmd)
    return cmds


# --------------------------------------------------------------------------
# Construction validation
# --------------------------------------------------------------------------

class TestValidation:
    def test_nominal_must_be_positive_finite(self):
        for bad in (0.0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError):
                GaitSpeedTracker(nominal_mps=bad)

    def test_negative_gains_rejected(self):
        with pytest.raises(ValueError):
            GaitSpeedTracker(nominal_mps=NOMINAL, kp=-0.1)
        with pytest.raises(ValueError):
            GaitSpeedTracker(nominal_mps=NOMINAL, ki=-1.0)

    def test_cadence_band_must_stay_inside_generator_clamp(self):
        # The trajectory generator ratio-clamps to [0.2, 2.0]; authority
        # outside it is a phantom stop.  Both edges enforced.
        with pytest.raises(ValueError):
            GaitSpeedTracker(nominal_mps=NOMINAL, max_cadence_scale=2.5)
        with pytest.raises(ValueError):
            GaitSpeedTracker(nominal_mps=NOMINAL, min_cadence_scale=0.1)
        with pytest.raises(ValueError):
            GaitSpeedTracker(
                nominal_mps=NOMINAL,
                min_cadence_scale=1.0, max_cadence_scale=0.5,
            )
        # The full legal band is accepted.
        GaitSpeedTracker(
            nominal_mps=NOMINAL, min_cadence_scale=0.2, max_cadence_scale=2.0,
        )

    def test_amp_floor_band(self):
        with pytest.raises(ValueError):
            GaitSpeedTracker(nominal_mps=NOMINAL, min_amp_scale=0.0)
        with pytest.raises(ValueError):
            GaitSpeedTracker(nominal_mps=NOMINAL, min_amp_scale=1.5)

    def test_slip_thresholds_positive(self):
        with pytest.raises(ValueError):
            GaitSpeedTracker(nominal_mps=NOMINAL, slip_probe_delta_mps=0.0)
        with pytest.raises(ValueError):
            GaitSpeedTracker(nominal_mps=NOMINAL, slip_tol_mps=-0.1)
        with pytest.raises(ValueError):
            GaitSpeedTracker(nominal_mps=NOMINAL, slip_release_s=0.0)

    def test_track_input_validation(self):
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        with pytest.raises(ValueError):
            tracker.track(0.0, 0.5, DT)      # a stop is not a setpoint
        with pytest.raises(ValueError):
            tracker.track(-1.0, 0.5, DT)
        with pytest.raises(ValueError):
            tracker.track(1.0, -0.5, DT)     # speed is a magnitude
        with pytest.raises(ValueError):
            tracker.track(1.0, 0.5, -0.1)
        with pytest.raises(ValueError):
            tracker.track(math.nan, 0.5, DT)
        with pytest.raises(ValueError):
            tracker.track(1.0, math.inf, DT)

    def test_integral_limit_derivation(self):
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        assert tracker.effective_integral_limit_mps == pytest.approx(
            tracker.max_demand_mps / tracker.ki
        )
        explicit = GaitSpeedTracker(nominal_mps=NOMINAL, integral_limit_mps=0.1)
        assert explicit.effective_integral_limit_mps == 0.1
        no_i = GaitSpeedTracker(nominal_mps=NOMINAL, ki=0.0)
        assert no_i.effective_integral_limit_mps == 0.0


# --------------------------------------------------------------------------
# The control case: inert when the setpoint is already achieved
# --------------------------------------------------------------------------

class TestInertWhenAchievable:
    def test_tracking_setpoint_is_a_byte_exact_passthrough(self):
        """measured == commanded -> demand IS commanded, amp untouched."""
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        commanded = 1.2
        state = None
        for _ in range(20):
            cmd = tracker.track(commanded, commanded, DT, state=state)
            state = cmd.state
            assert cmd.demand_mps == commanded          # ==, not approx
            assert cmd.amp_scale == 1.0
            assert cmd.error_mps == 0.0
            assert cmd.state.integral_mps == 0.0
            assert not cmd.at_ceiling
            assert not cmd.saturated
            assert not cmd.slip_latched
            assert cmd.shortfall_mps == 0.0
            assert cmd.ceiling_ticks == 0

    def test_small_error_inside_deadband_is_not_a_ceiling(self):
        """Noise-sized shortfall must not be reported as a wall."""
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL, deadband_mps=0.05)
        cmd = tracker.track(1.2, 1.17, DT)
        assert not cmd.at_ceiling
        # ... but the PI still works the small error (it is not a control
        # deadband): the demand moves above commanded.
        assert cmd.demand_mps > 1.2

    def test_convergence_to_an_achievable_setpoint(self):
        """Plant gain 0.5: needs demand 2x commanded — inside authority."""
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        commanded = 0.6  # needs demand 1.2 <= hi 2.0
        cmds = run_plant(tracker, commanded, lambda d: 0.5 * d, 60)
        final = cmds[-1]
        assert final.measured_mps == pytest.approx(commanded, abs=0.01)
        assert final.demand_mps == pytest.approx(commanded / 0.5, abs=0.05)
        assert not final.at_ceiling


# --------------------------------------------------------------------------
# Saturation: converge to the stop, do not wind up, report honestly
# --------------------------------------------------------------------------

class TestSaturationAndHonestCeiling:
    def test_unreachable_setpoint_converges_to_stop_without_windup(self):
        """Plant gain 0.3, commanded 1.5: needs demand 5.0, hi is 2.0."""
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL, slip_guard=False)
        hi = tracker.max_demand_mps
        cmds = run_plant(tracker, 1.5, lambda d: 0.3 * d, 60)

        final = cmds[-1]
        assert final.demand_mps == hi                  # parked at the stop
        assert final.at_ceiling
        assert final.saturated
        # The honest numbers: the body does 0.3 * 2.0 = 0.6 of the 1.5.
        assert final.measured_mps == pytest.approx(0.6, abs=1e-9)
        assert final.shortfall_mps == pytest.approx(0.9, abs=1e-9)
        # ceiling_ticks counts the steady condition, not a transient blip.
        assert final.ceiling_ticks > 20

        # Anti-windup: once parked, the integral must STOP MOVING — the
        # conditional-integration guarantee, checked across late ticks.
        late = [c.state.integral_mps for c in cmds[-10:]]
        assert len(set(late)) == 1
        limit = tracker.effective_integral_limit_mps
        assert all(abs(i) <= limit + 1e-12 for i in late)

    def test_recovery_after_saturation_has_no_windup_hangover(self):
        """Drop to an achievable setpoint: re-track fast, no long unwind."""
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL, slip_guard=False)
        plant = lambda d: 0.3 * d  # noqa: E731
        state = None
        demand = 1.5
        for _ in range(40):        # wind against the stop for 16 s
            cmd = tracker.track(1.5, plant(demand), DT, state=state)
            state, demand = cmd.state, cmd.demand_mps

        achievable = 0.5           # needs demand 1.667 — inside authority
        errors = []
        for _ in range(25):
            cmd = tracker.track(achievable, plant(demand), DT, state=state)
            state, demand = cmd.state, cmd.demand_mps
            errors.append(abs(cmd.error_mps))
        # Within 25 ticks (10 s) of the setpoint drop the loop is back on
        # target — a wound-up integrator would still be unwinding.
        assert errors[-1] < 0.03
        assert not cmd.at_ceiling

    def test_ceiling_ticks_resets_when_the_ceiling_lifts(self):
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL, slip_guard=False)
        state = None
        for _ in range(10):
            cmd = tracker.track(3.0, 0.4, DT, state=state)  # hopeless ask
            state = cmd.state
        assert cmd.ceiling_ticks == 10
        cmd = tracker.track(0.4, 0.4, DT, state=state)      # now achievable
        assert cmd.ceiling_ticks == 0
        assert not cmd.at_ceiling

    def test_stale_state_integral_is_entry_clamped(self):
        """A state from a different config cannot smuggle extra authority."""
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        limit = tracker.effective_integral_limit_mps
        stale = GaitSpeedState(integral_mps=50.0)
        cmd = tracker.track(1.0, 1.0, DT, state=stale)
        assert abs(cmd.state.integral_mps) <= limit + 1e-12
        assert cmd.demand_mps <= tracker.max_demand_mps


# --------------------------------------------------------------------------
# Authority limits hold under adversarial input
# --------------------------------------------------------------------------

class TestAuthorityLimits:
    def test_demand_never_leaves_the_band_and_amp_never_rises(self):
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        lo, hi = tracker.min_demand_mps, tracker.max_demand_mps
        adversarial = [
            (100.0, 0.0), (0.001, 50.0), (1.2, 0.0), (0.05, 3.0),
            (2.0, 2.0), (1.6, 0.001), (0.2, 0.2), (5.0, 0.0),
        ]
        state = None
        for commanded, measured in adversarial * 5:
            cmd = tracker.track(commanded, measured, DT, state=state)
            state = cmd.state
            assert lo <= cmd.demand_mps <= hi
            assert DEFAULT_MIN_AMP_SCALE <= cmd.amp_scale <= 1.0
            assert cmd.cadence_scale == pytest.approx(
                cmd.demand_mps / NOMINAL
            )
            assert DEFAULT_MIN_CADENCE_SCALE - 1e-12 <= cmd.cadence_scale
            assert cmd.cadence_scale <= DEFAULT_MAX_CADENCE_SCALE + 1e-12

    def test_amp_scale_is_strictly_downward_authority(self):
        """No input whatsoever may produce amp_scale > 1.0 — the direction
        measured to tumble the body is structurally unreachable."""
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        state = None
        for commanded, measured in (
            (3.0, 0.0),    # desperate shortfall — amp must NOT be raised
            (3.0, 0.1),
            (1.6, 1.59),
        ):
            cmd = tracker.track(commanded, measured, DT, state=state)
            state = cmd.state
            assert cmd.amp_scale == 1.0   # shortfall never raises amplitude

    def test_low_stop_overshoot_engages_downward_amp_trim(self):
        """Demand pinned at the floor, body still too fast: trim engages,
        proportional and floor-clamped."""
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        lo = tracker.min_demand_mps
        commanded, measured = 0.4 * lo, 2.0 * lo  # want 0.128, doing 0.64
        state = None
        for _ in range(5):
            cmd = tracker.track(commanded, measured, DT, state=state)
            state = cmd.state
        assert cmd.demand_mps == lo
        assert cmd.amp_scale == pytest.approx(
            max(DEFAULT_MIN_AMP_SCALE, commanded / measured)
        )
        assert cmd.amp_scale < 1.0
        # And the floor holds for an absurd overshoot.
        cmd = tracker.track(0.01, 5.0, DT, state=state)
        assert cmd.amp_scale == DEFAULT_MIN_AMP_SCALE

    def test_low_stop_freezes_the_integrator(self):
        """Winding DOWN against the floor is the same windup bug mirrored;
        the floor-side conditional integration must hold the integral."""
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        state = None
        integrals = []
        for _ in range(10):
            cmd = tracker.track(0.4, 1.0, DT, state=state)  # overshoot
            state = cmd.state
            integrals.append(cmd.state.integral_mps)
        # raw = 0.4 + (-0.6) + ki*I < lo from the first tick: frozen at 0.
        assert integrals == [0.0] * 10
        assert cmd.demand_mps == tracker.min_demand_mps


# --------------------------------------------------------------------------
# Slip guard: non-monotonic plant — converge below the knee, not past it
# --------------------------------------------------------------------------

def kneed_plant(knee: float, gain: float = 0.5, fall: float = 1.2):
    """Speed rises with demand to a knee, then FALLS — the measured shape
    (doubling cadence covered less ground)."""
    def plant(demand: float) -> float:
        if demand <= knee:
            return gain * demand
        return max(0.0, gain * knee - fall * (demand - knee))
    return plant


class TestSlipGuard:
    def test_latches_below_the_knee_instead_of_climbing_past_it(self):
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        knee = 1.4
        cmds = run_plant(tracker, 1.5, kneed_plant(knee), 20, demand0=1.5)
        latched = [c for c in cmds if c.slip_latched]
        assert latched, "guard never latched on a falling speed/demand curve"
        # Every latched tick's demand honours its own latched ceiling, and
        # no ceiling ever exceeds static authority.
        for c in latched:
            assert c.state.slip_ceiling_mps <= tracker.max_demand_mps
            assert c.demand_mps <= c.state.slip_ceiling_mps + 1e-9
        # The guard found an operating point NEAR the knee — below the
        # static stop the guard-less loop would park at (one slew step of
        # overshoot past the knee is the attribution resolution).
        assert min(c.state.slip_ceiling_mps for c in latched) <= knee + 0.31
        # And the ceiling report stays honest while latched short.
        assert any(c.at_ceiling for c in latched)

    def test_slow_gentle_decline_accumulates_to_a_latch(self):
        """The holding anchor catches a decline GENTLER than
        slip_tol/probe_delta per step — tick deltas alone never would."""
        tracker = GaitSpeedTracker(
            nominal_mps=NOMINAL, max_slew_mps_per_s=0.05,  # 0.02 per tick
        )
        cmds = run_plant(
            tracker, 1.5, kneed_plant(1.0, fall=0.5), 200,
            demand0=0.7,
            state=GaitSpeedState(active_demand_mps=0.7),  # warm start low
        )
        latched = [c for c in cmds if c.slip_latched]
        assert latched, "gentle decline never accumulated to a latch"
        # Latched at the base of the decline — near the knee, not far past.
        assert min(c.state.slip_ceiling_mps for c in latched) <= 1.1

    def test_latch_self_releases_and_reprobes_on_recovered_terrain(self):
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL, slip_release_s=2.0)
        # Phase 1: kneed plant until the guard latches.
        state = None
        demand = 1.5
        plant = kneed_plant(1.4)
        saw_latch = False
        for _ in range(6):
            cmd = tracker.track(1.5, plant(demand), DT, state=state)
            state, demand = cmd.state, cmd.demand_mps
            saw_latch = saw_latch or cmd.slip_latched
        assert saw_latch, "phase 1 never latched"
        latch_ceiling = state.slip_ceiling_mps
        # Phase 2: the slip condition clears (monotonic plant).  After
        # slip_release_s the latch must release and the demand climb PAST
        # the old ceiling instead of being crippled by a stale latch.
        released = False
        for _ in range(15):
            cmd = tracker.track(1.5, 0.5 * demand, DT, state=state)
            state, demand = cmd.state, cmd.demand_mps
            released = released or not cmd.slip_latched
        assert released, "slip latch never self-released"
        assert not cmd.slip_latched          # no re-latch on healthy terrain
        assert cmd.demand_mps > latch_ceiling + 0.05

    def test_guard_off_is_plain_saturation(self):
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL, slip_guard=False)
        cmds = run_plant(tracker, 1.5, kneed_plant(1.4), 30, demand0=1.5)
        assert not any(c.slip_latched for c in cmds)
        assert cmds[-1].demand_mps == tracker.max_demand_mps

    def test_monotonic_plant_never_latches(self):
        """A healthy plant must never trip the guard — no false ceiling on
        the plain saturation path."""
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        cmds = run_plant(tracker, 1.5, lambda d: 0.5 * d, 60)
        assert not any(c.slip_latched for c in cmds)


# --------------------------------------------------------------------------
# Determinism: same script, byte-identical command sequence
# --------------------------------------------------------------------------

class TestDeterminism:
    def test_replay_reproduces_the_command_stream(self):
        script = [
            (1.2, 0.45, 0.4), (1.2, 0.50, 0.4), (1.2, 0.48, 0.38),
            (0.6, 0.55, 0.4), (0.6, 0.58, 0.0),  # dt=0: integrator holds
            (2.0, 0.60, 0.4), (2.0, 0.55, 0.4), (0.4, 1.0, 0.4),
        ]

        def run():
            tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
            state = None
            rows = []
            for commanded, measured, dt in script:
                cmd = tracker.track(commanded, measured, dt, state=state)
                state = cmd.state
                rows.append(cmd.as_dict())
            return rows

        assert run() == run()

    def test_zero_dt_holds_the_integrator(self):
        tracker = GaitSpeedTracker(nominal_mps=NOMINAL)
        # A small unsaturated error, so the integral genuinely builds...
        s = tracker.track(1.2, 1.1, DT).state
        before = s.integral_mps
        assert before > 0.0
        # ...then a repeated timestamp must hold it exactly.
        cmd = tracker.track(1.2, 1.1, 0.0, state=s)
        assert cmd.state.integral_mps == before


# --------------------------------------------------------------------------
# StrideSpeedEstimator
# --------------------------------------------------------------------------

class TestStrideSpeedEstimator:
    def test_none_until_the_window_is_genuinely_spanned(self):
        est = StrideSpeedEstimator(window_s=1.0)
        assert est.update(0.0, 0.0, 0.0) is None
        assert est.update(0.5, 0.5, 0.0) is None      # half a window: None
        speed = est.update(1.0, 1.0, 0.0)             # spans exactly
        assert speed == pytest.approx(1.0)

    def test_constant_velocity_is_exact(self):
        est = StrideSpeedEstimator(window_s=0.8)
        speed = None
        for i in range(30):
            t = i * 0.1
            speed = est.update(t, 1.3 * t, -0.7 * t)
        assert speed == pytest.approx(math.hypot(1.3, -0.7))

    def test_stride_surge_averages_out_over_integer_strides(self):
        """The exact carrier that killed the StepReflex gate must vanish:
        mean speed 1.0 with a strong surge at 2.5 Hz, window = 2 strides."""
        hz = 2.5
        est = StrideSpeedEstimator(window_s=2.0 / hz)
        speed = None
        dt = 0.01
        for i in range(200):
            t = i * dt
            # x(t) = t + A*sin(2*pi*hz*t): instantaneous speed swings
            # 1 +/- 0.94, the windowed chord must still read ~1.0.
            x = t + 0.06 * math.sin(2.0 * math.pi * hz * t)
            speed = est.update(t, x, 0.0)
        assert speed == pytest.approx(1.0, abs=0.02)

    def test_time_reversal_raises_equal_time_ignored(self):
        est = StrideSpeedEstimator(window_s=1.0)
        est.update(0.0, 0.0, 0.0)
        est.update(2.0, 2.0, 0.0)
        with pytest.raises(ValueError):
            est.update(1.0, 1.0, 0.0)
        before = est.speed()
        assert est.update(2.0, 99.0, 99.0) == before  # paused clock: ignored

    def test_window_validation_and_reset(self):
        with pytest.raises(ValueError):
            StrideSpeedEstimator(window_s=0.0)
        with pytest.raises(ValueError):
            StrideSpeedEstimator(window_s=math.nan)
        est = StrideSpeedEstimator(window_s=0.5)
        est.update(0.0, 0.0, 0.0)
        est.update(1.0, 1.0, 0.0)
        assert est.speed() is not None
        est.reset()
        assert est.speed() is None

    def test_memory_stays_bounded_to_the_window(self):
        est = StrideSpeedEstimator(window_s=1.0)
        for i in range(10_000):
            est.update(i * 0.01, 0.0, 0.0)
        # window / dt + boundary sample + the newest
        assert len(est._samples) <= 102


# --------------------------------------------------------------------------
# GaitPhaseClock
# --------------------------------------------------------------------------

class TestGaitPhaseClock:
    def test_fixed_cadence_matches_t_times_hz(self):
        clock = GaitPhaseClock(2.6)
        for t in (0.0, 0.25, 1.0, 7.3):
            assert clock.phase_at(t) == pytest.approx(t * 2.6)

    def test_retime_is_phase_continuous(self):
        """The whole reason this class exists: no phase jump at a cadence
        change — the jump would kick all twelve joints at once."""
        clock = GaitPhaseClock(2.6)
        t_switch = 1.7
        before = clock.phase_at(t_switch)
        clock.retime(t_switch, 3.25)
        assert clock.phase_at(t_switch) == pytest.approx(before, rel=1e-12)
        # Slope changed to the new cadence after the switch...
        assert clock.phase_at(t_switch + 1.0) == pytest.approx(before + 3.25)
        # ...and repeated queries at one t are idempotent (pure lookup).
        assert clock.phase_at(2.0) == clock.phase_at(2.0)

    def test_chained_retimes_accumulate_continuously(self):
        clock = GaitPhaseClock(2.0)
        clock.retime(1.0, 4.0)    # phase 2.0 here
        clock.retime(2.0, 1.0)    # phase 6.0 here
        assert clock.phase_at(2.0) == pytest.approx(6.0)
        assert clock.phase_at(5.0) == pytest.approx(9.0)
        assert clock.stride_hz == 1.0

    def test_validation(self):
        with pytest.raises(ValueError):
            GaitPhaseClock(0.0)
        with pytest.raises(ValueError):
            GaitPhaseClock(-1.0)
        clock = GaitPhaseClock(2.0)
        with pytest.raises(ValueError):
            clock.retime(1.0, 0.0)
        with pytest.raises(ValueError):
            clock.retime(1.0, math.nan)


# --------------------------------------------------------------------------
# Composition sanity against the REAL trajectory generator (lib-side only)
# --------------------------------------------------------------------------

class TestComposesWithGaitCycle:
    def test_demand_stays_inside_the_generators_expressible_band(self):
        """Every demand the tracker can emit is a speed the generator will
        express without hitting its own silent ratio clamp."""
        from tritium_lib.models.quadruped import DEFAULT_GAITS

        spec = DEFAULT_GAITS["trot"]
        tracker = GaitSpeedTracker(nominal_mps=spec.speed_mps)
        lo_hz = 0.2 * spec.stride_hz
        hi_hz = 2.0 * spec.stride_hz
        state = None
        for commanded, measured in ((3.0, 0.0), (0.05, 2.0), (1.2, 0.45)):
            cmd = tracker.track(commanded, measured, DT, state=state)
            state = cmd.state
            from tritium_lib.models.gait_trajectory import QuadrupedGaitCycle

            cycle = QuadrupedGaitCycle("trot", speed=cmd.demand_mps)
            # The generator did NOT clamp: the demanded ratio was honest.
            assert cycle.speed_mps == pytest.approx(cmd.demand_mps)
            assert lo_hz - 1e-9 <= cycle.stride_hz <= hi_hz + 1e-9

    def test_phase_clock_drives_angles_without_a_jump_on_cadence_change(self):
        from tritium_lib.models.gait_trajectory import QuadrupedGaitCycle

        cycle = QuadrupedGaitCycle("trot", speed=1.2)
        clock = GaitPhaseClock(cycle.stride_hz)
        t_switch = 0.9
        before = cycle.angles_at_phase(clock.phase_at(t_switch))
        faster = QuadrupedGaitCycle("trot", speed=1.5)
        clock.retime(t_switch, faster.stride_hz)
        after = faster.angles_at_phase(clock.phase_at(t_switch))
        # Same phase, and thigh/calf angles continuous at the switch (amp
        # is identical at both speeds — the clamp pins it).
        for joint, angle in before.items():
            assert after[joint] == pytest.approx(angle, abs=1e-9)

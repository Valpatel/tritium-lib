# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""MatchReferee tests — transport-agnostic nerf-match duel scoring.

Covers:
  - Module-level dispersion_sigma == CombatSystem._dispersion_sigma (the
    extraction is byte-identical to the golden-calibrated staticmethod).
  - relative_fire_solution body-relative pan (dead ahead, due east, behind
    the shoulder / out of the servo arc) and distance.
  - Seeded determinism: identical rng seeds -> identical ShotOutcome
    sequences (golden-replay contract).
  - Miss gates: out_of_range (no rng draw, no damage), aim_off (>= 90 deg),
    geometric aim error pushing the lateral offset past the hit radius.
  - Statistical calibration: perfect aim at accuracy 0.8 lands ~80% of
    shots — the SAME calibration the projectile-flight sim proves.
  - Duel to KO: hp 40 vs damage 8 -> 5th hit destroys; winner()/active()/
    scoreboard() reflect the result.
  - spread_factor > 1 widens dispersion and lowers the hit rate.
"""

from __future__ import annotations

import random

import pytest

from tritium_lib.models.fire_control import PAN_MAX_DEG
from tritium_lib.models.hits import RegisterHitCommand
from tritium_lib.sim_engine.combat import (
    CombatSystem,
    HIT_RADIUS,
    MatchReferee,
    Weapon,
    dispersion_sigma,
    register_hit_command,
    relative_fire_solution,
)


# ---------------------------------------------------------------------------
# dispersion_sigma extraction parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("accuracy", [0.0, 0.5, 0.8, 0.95, 1.0])
def test_dispersion_sigma_matches_combat_system(accuracy: float) -> None:
    """The module-level function IS the staticmethod's math — identical out."""
    assert dispersion_sigma(accuracy) == CombatSystem._dispersion_sigma(accuracy)


# ---------------------------------------------------------------------------
# relative_fire_solution — body-relative pan
# ---------------------------------------------------------------------------

def test_relative_solution_dead_ahead() -> None:
    """Target straight down the chassis heading -> pan ~0, in arc."""
    sol, in_arc = relative_fire_solution((0.0, 0.0), 0.0, (0.0, 10.0))
    assert in_arc is True
    assert sol.pan == pytest.approx(0.0, abs=1e-9)
    assert sol.tilt == 0.0
    assert sol.distance == pytest.approx(10.0)


def test_relative_solution_due_east_of_north_facing() -> None:
    """North-facing shooter, target due east -> pan ~+90 (edge of arc)."""
    sol, in_arc = relative_fire_solution((0.0, 0.0), 0.0, (10.0, 0.0))
    assert in_arc is True
    assert sol.pan == pytest.approx(90.0)
    assert sol.distance == pytest.approx(10.0)


def test_relative_solution_behind_shoulder_out_of_arc() -> None:
    """Target behind the chassis: out of the servo arc, pan clamped."""
    sol, in_arc = relative_fire_solution((0.0, 0.0), 0.0, (0.0, -10.0))
    assert in_arc is False
    assert abs(sol.pan) <= PAN_MAX_DEG
    assert sol.distance == pytest.approx(10.0)


def test_relative_solution_heading_rotates_frame() -> None:
    """East-facing shooter, target north -> rel bearing -90 (left shoulder)."""
    sol, in_arc = relative_fire_solution((0.0, 0.0), 90.0, (0.0, 10.0))
    assert in_arc is True
    assert sol.pan == pytest.approx(-90.0)


def test_relative_solution_distance_is_hypot() -> None:
    sol, _ = relative_fire_solution((1.0, 2.0), 0.0, (4.0, 6.0))
    assert sol.distance == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Referee fixtures
# ---------------------------------------------------------------------------

def _duel(
    rng: random.Random,
    weapon_a: Weapon | None = None,
    weapon_b: Weapon | None = None,
    hp: float = 40.0,
) -> MatchReferee:
    """Two dogs 5m apart, both facing each other with centered turrets."""
    ref = MatchReferee(rng=rng)
    ref.add_combatant("dog_a", weapon=weapon_a, hp=hp)
    ref.add_combatant("dog_b", weapon=weapon_b, hp=hp)
    # dog_a at origin facing north; dog_b 5m north facing south (back at a).
    ref.update_pose("dog_a", 0.0, 0.0, 0.0, turret_pan_deg=0.0)
    ref.update_pose("dog_b", 0.0, 5.0, 180.0, turret_pan_deg=0.0)
    return ref


def _perfect_weapon(weapon_range: float = 20.0, damage: float = 8.0) -> Weapon:
    return Weapon(
        name="test_blaster", damage=damage, weapon_range=weapon_range,
        cooldown=1.0, accuracy=1.0, ammo=999, max_ammo=999,
    )


# ---------------------------------------------------------------------------
# Seeded determinism
# ---------------------------------------------------------------------------

def test_seeded_shot_sequences_identical() -> None:
    """Two referees seeded identically emit identical ShotOutcome sequences."""
    outcomes: list[list] = []
    for _ in range(2):
        ref = _duel(random.Random(42), hp=1e9)
        seq = [ref.resolve_shot("dog_a", "dog_b") for _ in range(50)]
        outcomes.append(seq)
    assert outcomes[0] == outcomes[1]


# ---------------------------------------------------------------------------
# Single-shot gates
# ---------------------------------------------------------------------------

def test_perfect_accuracy_perfect_aim_hits() -> None:
    ref = _duel(random.Random(42), weapon_a=_perfect_weapon())
    out = ref.resolve_shot("dog_a", "dog_b")
    assert out.hit is True
    assert out.reason == "hit"
    assert out.distance_m == pytest.approx(5.0)
    assert out.aim_error_deg == pytest.approx(0.0, abs=1e-9)
    assert out.damage_applied == 8.0
    assert out.target_hp_after == 32.0
    assert out.target_destroyed is False


def test_out_of_range_no_draw_no_damage() -> None:
    """Beyond weapon_range: miss with no rng draw and no damage applied."""
    rng = random.Random(7)
    ref = MatchReferee(rng=rng)
    ref.add_combatant("dog_a", weapon=_perfect_weapon(weapon_range=9.0))
    ref.add_combatant("dog_b")
    ref.update_pose("dog_a", 0.0, 0.0, 0.0)
    ref.update_pose("dog_b", 0.0, 50.0, 180.0)
    state_before = rng.getstate()
    out = ref.resolve_shot("dog_a", "dog_b")
    assert out.hit is False
    assert out.reason == "out_of_range"
    assert out.damage_applied == 0.0
    assert out.target_hp_after == 40.0
    assert rng.getstate() == state_before  # NO draw consumed
    assert ref.get_combatant("dog_a").shots_fired == 1  # trigger pull counted


def test_geometric_aim_error_misses() -> None:
    """45 deg off at 8m: lateral 8*tan(45)=8m > hit radius -> not a hit."""
    ref = MatchReferee(rng=random.Random(3))
    ref.add_combatant("dog_a", weapon=_perfect_weapon(weapon_range=20.0))
    ref.add_combatant("dog_b")
    ref.update_pose("dog_a", 0.0, 0.0, 0.0, turret_pan_deg=45.0)
    ref.update_pose("dog_b", 0.0, 8.0, 180.0)  # true bearing 0, aim 45
    out = ref.resolve_shot("dog_a", "dog_b")
    assert out.hit is False
    assert out.reason == "dispersion_miss"
    assert out.aim_error_deg == pytest.approx(45.0)
    assert abs(out.lateral_offset_m) > HIT_RADIUS


def test_aim_off_ninety_degrees_or_more() -> None:
    """Aim >= 90 deg off the true bearing: 'aim_off', no rng draw."""
    rng = random.Random(11)
    ref = MatchReferee(rng=rng)
    ref.add_combatant("dog_a", weapon=_perfect_weapon())
    ref.add_combatant("dog_b")
    # Facing south with centered turret; target due north -> error 180.
    ref.update_pose("dog_a", 0.0, 0.0, 180.0, turret_pan_deg=0.0)
    ref.update_pose("dog_b", 0.0, 5.0, 0.0)
    state_before = rng.getstate()
    out = ref.resolve_shot("dog_a", "dog_b")
    assert out.hit is False
    assert out.reason == "aim_off"
    assert abs(out.aim_error_deg) >= 90.0
    assert out.damage_applied == 0.0
    assert rng.getstate() == state_before


# ---------------------------------------------------------------------------
# Statistical calibration — the whole point of reusing dispersion_sigma
# ---------------------------------------------------------------------------

def test_hit_rate_matches_weapon_accuracy() -> None:
    """Perfect aim at accuracy 0.8 -> ~80% hit rate over 4000 shots."""
    weapon = Weapon(
        name="cal_blaster", damage=1.0, weapon_range=20.0,
        cooldown=1.0, accuracy=0.8, ammo=9999, max_ammo=9999,
    )
    ref = _duel(random.Random(1234), weapon_a=weapon, hp=1e9)
    shots = 4000
    hits = sum(
        1 for _ in range(shots) if ref.resolve_shot("dog_a", "dog_b").hit
    )
    rate = hits / shots
    assert 0.77 <= rate <= 0.83, f"hit rate {rate:.3f} outside calibration band"


def test_spread_factor_lowers_hit_rate() -> None:
    """spread_factor > 1 widens the dispersion -> measurably fewer hits."""
    weapon = Weapon(
        name="cal_blaster", damage=1.0, weapon_range=20.0,
        cooldown=1.0, accuracy=0.8, ammo=9999, max_ammo=9999,
    )
    rates: dict[float, float] = {}
    for spread in (1.0, 3.0):
        ref = _duel(random.Random(99), weapon_a=weapon, hp=1e9)
        shots = 2000
        hits = sum(
            1 for _ in range(shots)
            if ref.resolve_shot("dog_a", "dog_b", spread_factor=spread).hit
        )
        rates[spread] = hits / shots
    assert rates[3.0] < rates[1.0]


# ---------------------------------------------------------------------------
# Duel to KO
# ---------------------------------------------------------------------------

def test_duel_to_knockout_and_scoreboard() -> None:
    """hp 40 vs damage 8: the 5th landed hit destroys; referee calls it."""
    ref = _duel(random.Random(42), weapon_a=_perfect_weapon(damage=8.0))
    assert ref.active() is True
    assert ref.winner() is None

    last = None
    for _ in range(100):
        last = ref.resolve_shot("dog_a", "dog_b")
        if last.target_destroyed:
            break
    assert last is not None and last.target_destroyed is True
    assert last.hit is True
    assert last.target_hp_after == 0.0

    a = ref.get_combatant("dog_a")
    b = ref.get_combatant("dog_b")
    assert a.hits_landed == 5  # 40 / 8 = exactly 5 landed hits
    assert b.hits_taken == 5
    assert a.damage_dealt == 40.0
    assert b.damage_taken == 40.0
    assert b.alive is False

    assert ref.winner() == "dog_a"
    assert ref.active() is False

    board = ref.scoreboard()
    assert board["winner"] == "dog_a"
    assert set(board["combatants"]) == {"dog_a", "dog_b"}
    entry = board["combatants"]["dog_b"]
    assert entry["hp"] == 0.0
    assert entry["max_hp"] == 40.0
    assert entry["alive"] is False
    assert entry["hits_taken"] == 5
    assert entry["damage_taken"] == 40.0
    assert board["combatants"]["dog_a"]["weapon"] == "test_blaster"
    assert board["combatants"]["dog_a"]["shots_fired"] == a.shots_fired


def test_default_loadout_is_robot_dog() -> None:
    """add_combatant with no weapon equips the robot_dog default (own copy)."""
    ref = MatchReferee(rng=random.Random(0))
    a = ref.add_combatant("dog_a")
    b = ref.add_combatant("dog_b")
    assert a.weapon.name == "nerf_blaster"
    assert a.weapon.damage == 8.0
    assert a.weapon.weapon_range == 9.0
    assert a.weapon is not b.weapon  # fresh copies, no shared state


# ---------------------------------------------------------------------------
# Hit feedback — sync_health: the dog's reported health is authoritative
# ---------------------------------------------------------------------------

def test_sync_health_pins_referee_book_to_reported_hp() -> None:
    """Dog health telemetry overrides referee bookkeeping (wire matches)."""
    ref = _duel(random.Random(0))
    ref.sync_health("dog_b", 4.0)
    b = ref.get_combatant("dog_b")
    assert b.hp == 4.0
    assert b.alive is True
    ref.sync_health("dog_b", -3.0)  # nonsense negative report clamps to 0
    assert b.hp == 0.0
    assert b.alive is False


def test_sync_health_ko_resolves_on_reported_health() -> None:
    """A dog reporting hp 0 flips alive and decides the match."""
    ref = _duel(random.Random(0))
    assert ref.active() is True
    ref.sync_health("dog_b", 0.0)
    assert ref.get_combatant("dog_b").alive is False
    assert ref.winner() == "dog_a"
    assert ref.active() is False
    assert ref.scoreboard()["winner"] == "dog_a"


def test_sync_health_alive_false_forces_death() -> None:
    """A dog declaring itself dead is dead, whatever hp it claims."""
    ref = _duel(random.Random(0))
    ref.sync_health("dog_b", 12.0, alive=False)
    b = ref.get_combatant("dog_b")
    assert b.hp == 0.0
    assert b.alive is False


def test_sync_health_alive_true_leaves_hp_authoritative() -> None:
    ref = _duel(random.Random(0))
    ref.sync_health("dog_b", 7.5, alive=True)
    assert ref.get_combatant("dog_b").hp == 7.5


def test_sync_health_unknown_combatant_raises() -> None:
    ref = _duel(random.Random(0))
    with pytest.raises(KeyError):
        ref.sync_health("dog_zz", 10.0)


def test_sync_health_mirrors_reported_max_hp() -> None:
    """A dog whose own pool differs from the match seed corrects the book.

    The referee was seeded at the duel's default pool; a dog reporting its
    OWN max_hp (its config's hitpoints) should make the scoreboard show that
    pool, not the driver's guess — the dog owns its body.
    """
    ref = _duel(random.Random(0))
    ref.sync_health("dog_b", 8.0, max_hp=16.0)
    b = ref.get_combatant("dog_b")
    assert b.hp == 8.0
    assert b.max_hp == 16.0
    assert ref.scoreboard()["combatants"]["dog_b"]["max_hp"] == 16.0


def test_sync_health_max_hp_never_below_hp() -> None:
    """A nonsense max_hp below the reported hp is clamped up to hp."""
    ref = _duel(random.Random(0))
    ref.sync_health("dog_b", 20.0, max_hp=5.0)
    b = ref.get_combatant("dog_b")
    assert b.max_hp == 20.0  # clamped to hp, never shows less pool than hp


def test_sync_health_max_hp_none_leaves_seed() -> None:
    """Omitting max_hp leaves the seeded pool untouched (back-compat)."""
    ref = _duel(random.Random(0))
    seeded = ref.get_combatant("dog_b").max_hp
    ref.sync_health("dog_b", 7.0)
    assert ref.get_combatant("dog_b").max_hp == seeded


# ---------------------------------------------------------------------------
# Hit feedback — register_external_hit: hits the referee never adjudicated
# ---------------------------------------------------------------------------

def test_register_external_hit_books_both_sides() -> None:
    """Physical-sensor hit: target drained, shooter credited — but the
    trigger pull was already counted from ammo drain, so NO shots_fired."""
    ref = _duel(random.Random(0))
    hp_after = ref.register_external_hit("dog_b", 8.0, shooter_id="dog_a")
    assert hp_after == 32.0
    a = ref.get_combatant("dog_a")
    b = ref.get_combatant("dog_b")
    assert b.hp == 32.0
    assert b.hits_taken == 1
    assert b.damage_taken == 8.0
    assert a.hits_landed == 1
    assert a.damage_dealt == 8.0
    assert a.shots_fired == 0  # NOT incremented — already counted at fire time


def test_register_external_hit_unknown_or_none_shooter_safe() -> None:
    """A camera saw an impact but not who fired: target side books alone."""
    ref = _duel(random.Random(0))
    assert ref.register_external_hit("dog_b", 5.0, shooter_id="cam_ghost") == 35.0
    assert ref.register_external_hit("dog_b", 5.0, shooter_id=None) == 30.0
    b = ref.get_combatant("dog_b")
    assert b.hits_taken == 2
    assert b.damage_taken == 10.0
    a = ref.get_combatant("dog_a")
    assert a.hits_landed == 0
    assert a.damage_dealt == 0.0


def test_register_external_hit_clamps_at_zero_and_kos() -> None:
    ref = _duel(random.Random(0))
    hp_after = ref.register_external_hit("dog_b", 1000.0, shooter_id="dog_a")
    assert hp_after == 0.0
    assert ref.get_combatant("dog_b").alive is False
    assert ref.winner() == "dog_a"


def test_register_external_hit_unknown_target_raises() -> None:
    ref = _duel(random.Random(0))
    with pytest.raises(KeyError):
        ref.register_external_hit("dog_zz", 8.0)


# ---------------------------------------------------------------------------
# Hit feedback — register_hit_command: referee verdict -> wire command
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# N-dog free-for-all + deterministic multi-KO tie-break
# ---------------------------------------------------------------------------

def _ffa(rng: random.Random, ids: list[str], hp: float = 10.0) -> MatchReferee:
    """A free-for-all: N dogs, each its own team, strung out along +x."""
    ref = MatchReferee(rng=rng)
    for i, cid in enumerate(ids):
        ref.add_combatant(cid, hp=hp)
        ref.update_pose(cid, i * 3.0, 0.0, 0.0)
    return ref


def test_ffa_last_dog_standing_wins() -> None:
    """Four dogs enter, three fall in sequence; the survivor takes it."""
    ref = _ffa(random.Random(1), ["d1", "d2", "d3", "d4"])
    assert ref.active() is True
    assert ref.winner() is None
    # Drop three (in a scrambled order): d1 is the last dog standing.
    ref.sync_health("d2", 0.0)
    assert ref.active() is True  # three still stand -> contested
    ref.sync_health("d4", 0.0)
    ref.sync_health("d3", 0.0)
    assert ref.active() is False
    assert ref.winner() == "d1"
    assert ref.decide_winner() == "d1"
    assert ref.scoreboard()["decided_winner"] == "d1"
    assert ref.scoreboard()["winner"] == "d1"


def test_ffa_mid_match_is_contested_no_winner() -> None:
    """With two of four down, two teams still stand -> no decision yet."""
    ref = _ffa(random.Random(2), ["d1", "d2", "d3", "d4"])
    ref.sync_health("d1", 0.0)
    ref.sync_health("d2", 0.0)
    assert ref.active() is True
    assert ref.decide_winner() is None  # still contested
    assert ref.winner() is None


def test_defeated_seq_stamps_ko_order() -> None:
    """Each body is stamped with a strictly increasing KO-order number the
    first time it reaches 0 hp — the backbone of the tie-break."""
    ref = _ffa(random.Random(3), ["a", "b", "c"])
    assert all(ref.get_combatant(c).defeated_seq == 0 for c in ("a", "b", "c"))
    ref.sync_health("b", 0.0)
    ref.sync_health("a", 0.0)
    assert ref.get_combatant("b").defeated_seq == 1  # b fell first
    assert ref.get_combatant("a").defeated_seq == 2  # a fell second
    assert ref.get_combatant("c").defeated_seq == 0  # c never fell
    # Re-reporting a dead dog does NOT re-stamp it (seq is set once).
    ref.sync_health("b", 0.0)
    assert ref.get_combatant("b").defeated_seq == 1


def test_simultaneous_double_ko_tie_break_is_last_to_fall() -> None:
    """A true double-KO (both at 0 hp) is decided deterministically: the body
    that fell LAST wins — never dict/registration order.  Pinned against the
    seeded shot script (dog_a falls first, dog_b second)."""
    ref = MatchReferee(rng=random.Random(4242))
    ref.add_combatant("dog_a")
    ref.add_combatant("dog_b")
    ref.update_pose("dog_a", 0.0, 0.0, 0.0, turret_pan_deg=0.0)
    ref.update_pose("dog_b", 0.0, 5.0, 180.0, turret_pan_deg=0.0)
    for _ in range(10):
        ref.resolve_shot("dog_a", "dog_b")
        ref.resolve_shot("dog_b", "dog_a")
    a, b = ref.get_combatant("dog_a"), ref.get_combatant("dog_b")
    assert a.hp == 0.0 and b.hp == 0.0        # genuine double-KO
    assert a.defeated_seq == 1 and b.defeated_seq == 2  # a fell first
    assert ref.winner() is None                # sole-survivor notion: nobody
    assert ref.decide_winner() == "dog_b"      # fell last -> survived longest
    assert ref.scoreboard()["decided_winner"] == "dog_b"


def test_double_ko_tie_break_independent_of_registration_order() -> None:
    """The winner of a double-KO is the last to fall, whichever order the
    two were registered — the order-dependence bug is gone."""
    for first, second in (("x", "y"), ("y", "x")):
        ref = MatchReferee(rng=random.Random(0))
        ref.add_combatant(first, hp=8.0)
        ref.add_combatant(second, hp=8.0)
        ref.update_pose(first, 0.0, 0.0, 0.0)
        ref.update_pose(second, 0.0, 3.0, 180.0)
        ref.sync_health("x", 0.0)   # x always falls first
        ref.sync_health("y", 0.0)   # y always falls last -> y wins
        assert ref.decide_winner() == "y"


def test_teams_last_team_standing() -> None:
    """A team match resolves to the surviving team; the winner id is that
    team's healthiest survivor."""
    ref = MatchReferee(rng=random.Random(5))
    for cid, team, hp in (("r1", "red", 10.0), ("r2", "red", 4.0),
                          ("b1", "blue", 10.0), ("b2", "blue", 10.0)):
        ref.add_combatant(cid, hp=hp, team=team)
        ref.update_pose(cid, 0.0, 0.0, 0.0)
    assert ref.active() is True
    ref.sync_health("b1", 0.0)
    assert ref.active() is True   # blue still has b2
    ref.sync_health("b2", 0.0)
    assert ref.active() is False  # red wins — both blue down
    assert ref.winning_team() == "red"
    assert ref.decide_winner() == "r1"   # red's healthiest (10 > 4)
    assert ref.scoreboard()["winning_team"] == "red"
    assert ref.scoreboard()["combatants"]["r1"]["team"] == "red"


def test_decide_winner_needs_two_combatants() -> None:
    """A lone entrant is not a match — no winner is decided."""
    ref = MatchReferee(rng=random.Random(0))
    ref.add_combatant("solo")
    assert ref.decide_winner() is None
    assert ref.winning_team() is None


def test_register_hit_command_from_resolved_outcome() -> None:
    """A seeded referee hit maps onto the exact wire command the target
    dog's brain keys on."""
    ref = _duel(random.Random(42), weapon_a=_perfect_weapon(damage=8.0))
    out = ref.resolve_shot("dog_a", "dog_b")
    assert out.hit is True  # deterministic under seed 42 (perfect weapon)
    cmd = register_hit_command(out)
    assert isinstance(cmd, RegisterHitCommand)
    assert cmd.command == "register_hit"
    assert cmd.shooter_id == "dog_a"
    assert cmd.damage == out.damage_applied == 8.0
    assert cmd.source == "referee"
    assert len(cmd.hit_id) == 12
    assert cmd.model_dump()["command"] == "register_hit"  # wire keyable


# ---------------------------------------------------------------------------
# Regression — the hit-feedback additions must not shift the rng sequence
# ---------------------------------------------------------------------------

def test_seeded_shot_script_pinned_values() -> None:
    """Fixed pose + shot script under seed 4242: every number below was
    pinned by running once and must stay byte-stable across runs — proof
    the additive hit-feedback methods leave resolve_shot's draw sequence
    untouched (golden-replay contract)."""
    ref = MatchReferee(rng=random.Random(4242))
    ref.add_combatant("dog_a")  # default robot_dog nerf_blaster
    ref.add_combatant("dog_b")
    ref.update_pose("dog_a", 0.0, 0.0, 0.0, turret_pan_deg=0.0)
    ref.update_pose("dog_b", 0.0, 5.0, 180.0, turret_pan_deg=0.0)

    outcomes = []
    for _ in range(10):
        outcomes.append(ref.resolve_shot("dog_a", "dog_b"))
        outcomes.append(ref.resolve_shot("dog_b", "dog_a"))

    assert [o.reason for o in outcomes] == [
        "hit", "hit", "hit", "hit", "hit", "hit",
        "dispersion_miss", "hit", "hit", "hit",
        "hit", "hit", "hit", "hit", "hit", "hit",
        "hit", "dispersion_miss", "hit", "hit",
    ]
    assert [o.target_hp_after for o in outcomes] == [
        32.0, 32.0, 24.0, 24.0, 16.0, 16.0, 16.0, 8.0, 8.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    assert outcomes[0].lateral_offset_m == pytest.approx(
        2.62576998717928, abs=1e-12)
    assert outcomes[1].lateral_offset_m == pytest.approx(
        -3.0763062233690426, abs=1e-12)
    assert outcomes[2].lateral_offset_m == pytest.approx(
        3.819054888004615, abs=1e-12)
    assert outcomes[3].lateral_offset_m == pytest.approx(
        0.6900804233353119, abs=1e-12)

    for cid in ("dog_a", "dog_b"):
        c = ref.get_combatant(cid)
        assert c.hp == 0.0
        assert c.shots_fired == 10
        assert c.hits_landed == 9
        assert c.hits_taken == 9
        assert c.damage_dealt == 72.0
        assert c.damage_taken == 72.0
        assert c.alive is False

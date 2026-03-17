"""Tests for tritium_lib.sim_engine.audio.spatial — spatial audio math."""
import math
import pytest

from tritium_lib.sim_engine.audio.spatial import (
    SoundEvent,
    distance_attenuation,
    doppler_factor,
    explosion_parameters,
    gunshot_layers,
    occlusion_factor,
    propagation_delay,
    reverb_level,
    stereo_pan,
)


# ---------------------------------------------------------------------------
# distance_attenuation
# ---------------------------------------------------------------------------

class TestDistanceAttenuation:
    def test_at_ref_distance_is_one(self):
        gain = distance_attenuation((0, 0), (1, 0), ref_distance=1.0)
        assert gain == pytest.approx(1.0)

    def test_falls_off_with_distance(self):
        g_near = distance_attenuation((0, 0), (10, 0))
        g_far = distance_attenuation((0, 0), (100, 0))
        assert g_near > g_far > 0.0

    def test_at_zero_distance(self):
        gain = distance_attenuation((5, 5), (5, 5))
        assert gain == pytest.approx(1.0)

    def test_clamped_to_max_distance(self):
        g1 = distance_attenuation((0, 0), (500, 0), max_distance=500.0)
        g2 = distance_attenuation((0, 0), (1000, 0), max_distance=500.0)
        assert g1 == pytest.approx(g2)

    def test_higher_rolloff_decays_faster(self):
        g_low = distance_attenuation((0, 0), (50, 0), rolloff=0.5)
        g_high = distance_attenuation((0, 0), (50, 0), rolloff=2.0)
        assert g_low > g_high

    def test_returns_between_zero_and_one(self):
        for d in [0, 1, 10, 100, 500, 1000]:
            g = distance_attenuation((0, 0), (d, 0))
            assert 0.0 <= g <= 1.0


# ---------------------------------------------------------------------------
# stereo_pan
# ---------------------------------------------------------------------------

class TestStereoPan:
    def test_source_directly_right_positive_pan(self):
        """Listener facing north (+Y), source to the east (+X) = right = positive."""
        pan = stereo_pan((10, 0), (0, 0), listener_heading=0.0)
        assert pan > 0.0

    def test_source_directly_left_negative_pan(self):
        """Source to the west (-X) = left = negative."""
        pan = stereo_pan((-10, 0), (0, 0), listener_heading=0.0)
        assert pan < 0.0

    def test_source_directly_ahead_zero_pan(self):
        """Source directly ahead should be ~0 pan."""
        pan = stereo_pan((0, 10), (0, 0), listener_heading=0.0)
        assert pan == pytest.approx(0.0, abs=0.01)

    def test_source_directly_behind_zero_pan(self):
        """Source directly behind should be ~0 pan."""
        pan = stereo_pan((0, -10), (0, 0), listener_heading=0.0)
        assert pan == pytest.approx(0.0, abs=0.01)

    def test_same_position_zero_pan(self):
        pan = stereo_pan((5, 5), (5, 5), listener_heading=1.0)
        assert pan == 0.0

    def test_heading_rotates_pan(self):
        """Turning right should shift a forward source to the left."""
        # Source at (0, 10), listener at origin facing north -> ~0 pan
        pan_north = stereo_pan((0, 10), (0, 0), listener_heading=0.0)
        # Listener turns to face east (heading = pi/2) -> source now to the left
        pan_east = stereo_pan((0, 10), (0, 0), listener_heading=math.pi / 2)
        assert pan_east < pan_north

    def test_pan_range(self):
        """Pan should always be in [-1, 1]."""
        for angle in range(0, 360, 15):
            rad = math.radians(angle)
            x = 10 * math.cos(rad)
            y = 10 * math.sin(rad)
            pan = stereo_pan((x, y), (0, 0), listener_heading=0.0)
            assert -1.0 <= pan <= 1.0


# ---------------------------------------------------------------------------
# doppler_factor
# ---------------------------------------------------------------------------

class TestDopplerFactor:
    def test_approaching_source_pitch_up(self):
        """Source moving toward listener -> factor > 1.0."""
        factor = doppler_factor(
            source_vel=(0, 10),   # moving toward listener
            listener_vel=(0, 0),
            source_pos=(0, 0),
            listener_pos=(0, 100),
        )
        assert factor > 1.0

    def test_receding_source_pitch_down(self):
        """Source moving away from listener -> factor < 1.0."""
        factor = doppler_factor(
            source_vel=(0, -10),  # moving away
            listener_vel=(0, 0),
            source_pos=(0, 0),
            listener_pos=(0, 100),
        )
        assert factor < 1.0

    def test_stationary_is_one(self):
        factor = doppler_factor((0, 0), (0, 0), (0, 0), (0, 100))
        assert factor == pytest.approx(1.0)

    def test_same_velocity_is_one(self):
        """Both moving same direction at same speed -> no shift."""
        factor = doppler_factor((5, 5), (5, 5), (0, 0), (0, 100))
        assert factor == pytest.approx(1.0, abs=0.01)

    def test_same_position_is_one(self):
        factor = doppler_factor((10, 0), (0, 0), (5, 5), (5, 5))
        assert factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# propagation_delay
# ---------------------------------------------------------------------------

class TestPropagationDelay:
    def test_343m_is_one_second(self):
        delay = propagation_delay((0, 0), (343, 0), speed_of_sound=343.0)
        assert delay == pytest.approx(1.0)

    def test_zero_distance_zero_delay(self):
        delay = propagation_delay((5, 5), (5, 5))
        assert delay == pytest.approx(0.0)

    def test_scales_linearly(self):
        d1 = propagation_delay((0, 0), (100, 0))
        d2 = propagation_delay((0, 0), (200, 0))
        assert d2 == pytest.approx(2 * d1)


# ---------------------------------------------------------------------------
# occlusion_factor
# ---------------------------------------------------------------------------

class TestOcclusionFactor:
    def test_no_obstacles_clear(self):
        assert occlusion_factor((0, 0), (100, 0), []) == pytest.approx(1.0)

    def test_obstacle_between_reduces_gain(self):
        """Wall in the middle blocks some sound."""
        obstacles = [((50, 0), 5.0)]  # Circle at midpoint, radius 5
        factor = occlusion_factor((0, 0), (100, 0), obstacles)
        assert factor < 1.0

    def test_obstacle_off_path_no_effect(self):
        """Obstacle far from the line of sight doesn't block."""
        obstacles = [((50, 100), 5.0)]  # Far off to the side
        factor = occlusion_factor((0, 0), (100, 0), obstacles)
        assert factor == pytest.approx(1.0)

    def test_multiple_obstacles_stack(self):
        one = [((30, 0), 5.0)]
        two = [((30, 0), 5.0), ((70, 0), 5.0)]
        f1 = occlusion_factor((0, 0), (100, 0), one)
        f2 = occlusion_factor((0, 0), (100, 0), two)
        assert f2 < f1

    def test_fully_blocked_approaches_zero(self):
        """Many obstacles drive factor toward zero."""
        obstacles = [((i * 10, 0), 3.0) for i in range(1, 20)]
        factor = occlusion_factor((0, 0), (200, 0), obstacles)
        assert factor < 0.05


# ---------------------------------------------------------------------------
# reverb_level
# ---------------------------------------------------------------------------

class TestReverbLevel:
    def test_no_buildings_dry(self):
        assert reverb_level((0, 0), []) == pytest.approx(0.0)

    def test_nearby_buildings_add_reverb(self):
        buildings = [((10, 0), 5.0), ((-10, 0), 5.0), ((0, 10), 5.0)]
        level = reverb_level((0, 0), buildings)
        assert level > 0.0

    def test_far_buildings_less_reverb(self):
        near = [((10, 0), 5.0)]
        far = [((90, 0), 5.0)]
        r_near = reverb_level((0, 0), near)
        r_far = reverb_level((0, 0), far)
        assert r_near > r_far

    def test_capped_at_one(self):
        """Even many buildings shouldn't exceed 1.0."""
        buildings = [((i, j), 5.0) for i in range(-50, 51, 10) for j in range(-50, 51, 10)]
        level = reverb_level((0, 0), buildings)
        assert level <= 1.0


# ---------------------------------------------------------------------------
# gunshot_layers
# ---------------------------------------------------------------------------

class TestGunshotLayers:
    def test_crack_before_blast(self):
        """Supersonic crack arrives before muzzle blast at distance."""
        layers = gunshot_layers(500.0)
        assert layers["supersonic_crack_delay"] < layers["muzzle_blast_delay"]

    def test_echo_after_blast(self):
        layers = gunshot_layers(200.0)
        assert layers["echo_delay"] > layers["muzzle_blast_delay"]

    def test_has_all_keys(self):
        layers = gunshot_layers(100.0)
        assert "muzzle_blast_delay" in layers
        assert "supersonic_crack_delay" in layers
        assert "echo_delay" in layers
        assert "gain" in layers
        assert "distance" in layers

    def test_gain_decreases_with_distance(self):
        g_near = gunshot_layers(50.0)["gain"]
        g_far = gunshot_layers(500.0)["gain"]
        assert g_near > g_far


# ---------------------------------------------------------------------------
# explosion_parameters
# ---------------------------------------------------------------------------

class TestExplosionParameters:
    def test_has_all_keys(self):
        params = explosion_parameters(100.0)
        for key in ("gain", "low_freq_boost", "duration", "shake", "delay", "yield_kg"):
            assert key in params

    def test_bigger_yield_more_bass(self):
        small = explosion_parameters(100.0, yield_kg=1.0)
        big = explosion_parameters(100.0, yield_kg=100.0)
        assert big["low_freq_boost"] > small["low_freq_boost"]

    def test_bigger_yield_longer_duration(self):
        small = explosion_parameters(100.0, yield_kg=1.0)
        big = explosion_parameters(100.0, yield_kg=50.0)
        assert big["duration"] > small["duration"]

    def test_shake_decreases_with_distance(self):
        near = explosion_parameters(10.0, yield_kg=5.0)
        far = explosion_parameters(500.0, yield_kg=5.0)
        assert near["shake"] > far["shake"]

    def test_shake_capped_at_one(self):
        params = explosion_parameters(1.0, yield_kg=1000.0)
        assert params["shake"] <= 1.0


# ---------------------------------------------------------------------------
# SoundEvent
# ---------------------------------------------------------------------------

class TestSoundEvent:
    def test_compute_for_listener_returns_valid_dict(self):
        event = SoundEvent("gunshot_01", (100, 50), volume=0.8, category="effect")
        result = event.compute_for_listener(
            listener_pos=(0, 0),
            listener_heading=0.0,
        )
        assert isinstance(result, dict)
        assert "sound_id" in result
        assert result["sound_id"] == "gunshot_01"
        assert "gain" in result
        assert "pan" in result
        assert "delay" in result
        assert "pitch" in result
        assert "reverb" in result
        assert "category" in result
        assert result["category"] == "effect"

    def test_gain_includes_volume(self):
        """Event volume should scale the output gain."""
        loud = SoundEvent("boom", (50, 0), volume=1.0)
        quiet = SoundEvent("boom", (50, 0), volume=0.5)
        r_loud = loud.compute_for_listener((0, 0))
        r_quiet = quiet.compute_for_listener((0, 0))
        assert r_loud["gain"] > r_quiet["gain"]

    def test_obstacles_reduce_gain(self):
        event = SoundEvent("step", (100, 0))
        clear = event.compute_for_listener((0, 0))
        blocked = event.compute_for_listener((0, 0), obstacles=[((50, 0), 5.0)])
        assert blocked["gain"] < clear["gain"]

    def test_to_dict(self):
        event = SoundEvent("alert", (10, 20), volume=0.9, pitch=1.2, category="voice")
        d = event.to_dict()
        assert d["sound_id"] == "alert"
        assert d["position"] == [10, 20]
        assert d["volume"] == 0.9
        assert d["pitch"] == 1.2
        assert d["category"] == "voice"

    def test_doppler_applied_to_pitch(self):
        """Moving source should shift pitch."""
        event = SoundEvent("siren", (0, 0), pitch=1.0)
        # Source approaching
        result = event.compute_for_listener(
            listener_pos=(0, 100),
            source_vel=(0, 10),
        )
        assert result["pitch"] > 1.0

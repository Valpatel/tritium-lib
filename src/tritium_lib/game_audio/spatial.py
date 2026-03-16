"""Spatial audio math for positioning sounds on the tactical map.

These functions compute audio parameters that the frontend Web Audio API
(PannerNode / Three.js PositionalAudio) uses for playback. The backend
computes the math, the browser plays the sound.
"""
import math

from tritium_lib.game_debug.streams import DebugStream

Vec2 = tuple[float, float]


def _dist(a: Vec2, b: Vec2) -> float:
    """Euclidean distance between two 2D points."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.sqrt(dx * dx + dy * dy)


def _dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def _length(v: Vec2) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1])


def distance_attenuation(source: Vec2, listener: Vec2,
                         ref_distance: float = 1.0,
                         max_distance: float = 500.0,
                         rolloff: float = 1.0) -> float:
    """Inverse distance attenuation following the Web Audio API model.

    Uses the inverse distance model:
        gain = ref_distance / (ref_distance + rolloff * (clamp(d, ref, max) - ref))

    Returns gain 0.0-1.0.
    """
    d = _dist(source, listener)
    d = max(d, ref_distance)
    d = min(d, max_distance)
    denom = ref_distance + rolloff * (d - ref_distance)
    if denom <= 0:
        return 0.0
    return min(1.0, max(0.0, ref_distance / denom))


def stereo_pan(source: Vec2, listener: Vec2, listener_heading: float) -> float:
    """Compute stereo pan -1.0 (left) to 1.0 (right) based on relative angle.

    *listener_heading* is in **radians**, measured clockwise from +Y (north).
    A heading of 0 means the listener faces north (+Y direction).

    Convention: +X is east/right when facing north.
    """
    dx = source[0] - listener[0]
    dy = source[1] - listener[1]

    if dx == 0.0 and dy == 0.0:
        return 0.0

    # Angle from listener to source in world space (atan2 gives CCW from +X)
    angle_to_source = math.atan2(dy, dx)

    # Convert listener heading to math convention: heading 0 = +Y = pi/2 in atan2
    listener_angle = math.pi / 2.0 - listener_heading

    # Relative angle: positive means source is to the right
    relative = angle_to_source - listener_angle
    # Normalize to [-pi, pi]
    relative = math.atan2(math.sin(relative), math.cos(relative))

    # Pan: sin of relative angle gives left/right.
    # When relative > 0 (source CCW from facing dir in math coords),
    # that means source is to the LEFT in heading convention.
    # So negate to get: positive = right.
    pan = -math.sin(relative)
    return max(-1.0, min(1.0, pan))


def doppler_factor(source_vel: Vec2, listener_vel: Vec2,
                   source_pos: Vec2, listener_pos: Vec2,
                   speed_of_sound: float = 343.0) -> float:
    """Doppler pitch shift factor.

    > 1.0 = source approaching listener (pitch up).
    < 1.0 = source receding from listener (pitch down).

    Uses the classic Doppler formula:
        f'/f = (c - v_listener_toward) / (c - v_source_toward)
    where v_toward is the velocity component along the source→listener axis.
    """
    diff = _sub(listener_pos, source_pos)
    d = _length(diff)
    if d < 1e-9:
        return 1.0

    # Unit vector from source to listener
    n = (diff[0] / d, diff[1] / d)

    # Project velocities onto source→listener axis
    v_source_toward = _dot(source_vel, n)     # positive = moving toward listener
    v_listener_toward = _dot(listener_vel, n)  # positive = moving toward source

    # Doppler formula: listener moving toward raises pitch, source moving toward raises pitch
    denom = speed_of_sound - v_source_toward
    numer = speed_of_sound - v_listener_toward

    if abs(denom) < 1e-9:
        return 1.0

    factor = numer / denom
    # Clamp to reasonable range
    return max(0.1, min(10.0, factor))


def propagation_delay(source: Vec2, listener: Vec2,
                      speed_of_sound: float = 343.0) -> float:
    """Time delay in seconds for sound to travel from source to listener."""
    d = _dist(source, listener)
    if speed_of_sound <= 0:
        return 0.0
    return d / speed_of_sound


def _ray_circle_intersect(ray_origin: Vec2, ray_dir: Vec2,
                          circle_center: Vec2, circle_radius: float,
                          ray_length: float) -> bool:
    """Test if a ray segment intersects a circle."""
    # Vector from ray origin to circle center
    oc = _sub(circle_center, ray_origin)
    # Project onto ray direction
    dir_len = _length(ray_dir)
    if dir_len < 1e-12:
        return False
    nd = (ray_dir[0] / dir_len, ray_dir[1] / dir_len)

    proj = _dot(oc, nd)
    # Closest point distance squared
    oc_len_sq = oc[0] * oc[0] + oc[1] * oc[1]
    closest_dist_sq = oc_len_sq - proj * proj

    if closest_dist_sq < 0:
        closest_dist_sq = 0.0

    r_sq = circle_radius * circle_radius
    if closest_dist_sq > r_sq:
        return False

    # Check the intersection is within the ray segment
    half_chord = math.sqrt(max(0.0, r_sq - closest_dist_sq))
    t_enter = proj - half_chord
    t_exit = proj + half_chord

    # Intersection if any part of [t_enter, t_exit] overlaps [0, ray_length]
    return t_exit >= 0 and t_enter <= ray_length


def occlusion_factor(source: Vec2, listener: Vec2,
                     obstacles: list[tuple[Vec2, float]]) -> float:
    """How much sound is blocked by obstacles. 1.0 = clear, 0.0 = fully blocked.

    Each obstacle is (center, radius). Uses ray-circle intersection.
    Each blocking obstacle reduces the factor by 0.3 (stacks multiplicatively).
    """
    if not obstacles:
        return 1.0

    diff = _sub(listener, source)
    d = _length(diff)
    if d < 1e-9:
        return 1.0

    factor = 1.0
    for center, radius in obstacles:
        if _ray_circle_intersect(source, diff, center, radius, d):
            factor *= 0.7  # Each obstacle lets 70% through

    return max(0.0, factor)


def reverb_level(source: Vec2, buildings: list[tuple[Vec2, float]],
                 max_range: float = 100.0) -> float:
    """Estimate reverb amount based on nearby building density.

    More buildings within *max_range* of the source = more echo.
    Open field = dry (0.0). Dense urban = wet (up to 1.0).

    Each building contributes based on proximity: closer buildings reflect more.
    """
    if not buildings:
        return 0.0

    total = 0.0
    for center, radius in buildings:
        d = _dist(source, center) - radius
        d = max(0.0, d)
        if d < max_range:
            # Closer buildings contribute more reverb
            contribution = 1.0 - (d / max_range)
            total += contribution * 0.15  # Each building adds up to 15%

    return min(1.0, total)


class SoundEvent:
    """A sound that should be played at a position on the map."""

    # Class-level debug stream shared across all SoundEvent instances
    debug = DebugStream("audio")

    def __init__(self, sound_id: str, position: Vec2,
                 volume: float = 1.0, pitch: float = 1.0,
                 category: str = "effect"):
        self.sound_id = sound_id
        self.position = position
        self.volume = volume
        self.pitch = pitch
        self.category = category  # "effect", "ambient", "voice", "music"

    def compute_for_listener(self, listener_pos: Vec2,
                             listener_heading: float = 0.0,
                             obstacles: list[tuple[Vec2, float]] | None = None,
                             buildings: list[tuple[Vec2, float]] | None = None,
                             listener_vel: Vec2 = (0.0, 0.0),
                             source_vel: Vec2 = (0.0, 0.0)) -> dict:
        """Compute all audio parameters for a specific listener position.

        Returns dict ready for WebSocket -> frontend Web Audio API.
        """
        occ = occlusion_factor(self.position, listener_pos, obstacles or [])
        gain = distance_attenuation(self.position, listener_pos) * self.volume * occ

        result = {
            "sound_id": self.sound_id,
            "gain": round(gain, 4),
            "pan": round(stereo_pan(self.position, listener_pos, listener_heading), 4),
            "delay": round(propagation_delay(self.position, listener_pos), 4),
            "pitch": round(self.pitch * doppler_factor(source_vel, listener_vel,
                                                       self.position, listener_pos), 4),
            "reverb": round(reverb_level(self.position, buildings or []), 4),
            "category": self.category,
        }

        # Emit debug data
        if SoundEvent.debug.enabled:
            frame = SoundEvent.debug.begin_frame()
            if frame is not None:
                frame.entries.append({
                    "type": "sound_computed",
                    "sound_id": self.sound_id,
                    "source_pos": list(self.position),
                    "listener_pos": list(listener_pos),
                    "gain": result["gain"],
                    "pan": result["pan"],
                    "delay": result["delay"],
                    "pitch": result["pitch"],
                    "reverb": result["reverb"],
                    "occlusion": round(occ, 4),
                })
                SoundEvent.debug.end_frame(frame)

        return result

    def to_dict(self) -> dict:
        """Serialize for transport."""
        return {
            "sound_id": self.sound_id,
            "position": list(self.position),
            "volume": self.volume,
            "pitch": self.pitch,
            "category": self.category,
        }


# ---------------------------------------------------------------------------
# Combat-specific helpers
# ---------------------------------------------------------------------------

def gunshot_layers(distance: float) -> dict:
    """Compute timing for layered gunshot sound at given distance.

    A gunshot produces three distinct sounds:
    1. **Muzzle blast** — travels at speed of sound (343 m/s)
    2. **Supersonic crack** — bullet travels ~900 m/s, crack radiates at 343 m/s
       but originates from the bullet's path, arriving before the muzzle blast
       if the listener is downrange
    3. **Echo** — reflected muzzle blast, arrives ~0.3-2s after blast

    Returns delays in seconds from the moment of firing.
    """
    speed_of_sound = 343.0
    bullet_speed = 900.0  # Typical supersonic rifle round

    muzzle_delay = distance / speed_of_sound
    # Crack arrives earlier because the bullet is closer when it passes
    # Approximation: bullet reaches listener vicinity, then crack propagates short distance
    if distance > 0 and bullet_speed > speed_of_sound:
        bullet_travel = distance / bullet_speed
        # Crack radiates from bullet position — minimal extra travel
        crack_delay = bullet_travel + 0.01  # Small offset for lateral propagation
    else:
        crack_delay = muzzle_delay

    # Echo: muzzle blast bouncing off surfaces, typically 1.5x-3x the direct path
    echo_delay = muzzle_delay * 2.0 + 0.1  # Simplified: double-path + constant

    # Gain falls off with distance
    gain = distance_attenuation((0, 0), (distance, 0), ref_distance=5.0,
                                max_distance=2000.0, rolloff=1.0)

    return {
        "muzzle_blast_delay": round(muzzle_delay, 4),
        "supersonic_crack_delay": round(crack_delay, 4),
        "echo_delay": round(echo_delay, 4),
        "gain": round(gain, 4),
        "distance": distance,
    }


def explosion_parameters(distance: float, yield_kg: float = 1.0) -> dict:
    """Compute explosion sound parameters.

    Returns:
        gain: overall volume attenuation
        low_freq_boost: bass emphasis (dB) — bigger explosions are bassier
        duration: how long the sound lasts (seconds)
        shake: screen shake intensity 0.0-1.0
        delay: propagation delay in seconds
    """
    speed_of_sound = 343.0
    delay = distance / speed_of_sound if distance > 0 else 0.0

    # Gain: explosions are louder, use larger ref_distance
    ref_dist = 10.0 * math.sqrt(yield_kg)
    gain = distance_attenuation((0, 0), (distance, 0),
                                ref_distance=ref_dist,
                                max_distance=5000.0,
                                rolloff=0.8)

    # Low frequency boost scales with yield
    low_freq_boost = min(24.0, 6.0 * math.log2(max(1.0, yield_kg)) + 6.0)

    # Duration: bigger boom = longer rumble
    duration = min(10.0, 0.5 + 0.3 * math.sqrt(yield_kg))

    # Screen shake: falls off with distance, scales with yield
    shake_raw = (yield_kg ** 0.4) / (1.0 + distance / 50.0)
    shake = min(1.0, max(0.0, shake_raw))

    return {
        "gain": round(gain, 4),
        "low_freq_boost": round(low_freq_boost, 2),
        "duration": round(duration, 3),
        "shake": round(shake, 4),
        "delay": round(delay, 4),
        "distance": distance,
        "yield_kg": yield_kg,
    }

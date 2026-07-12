# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Synthetic baseband-IQ generator for Mode S / ADS-B (1090 MHz).

RF is sensor priority #1, but the whole decode chain (preamble correlation,
112-bit pulse-position-modulation bit-slice, CRC-24 validation, CPR position
math) is impossible to exercise without a radio in the loop.  This module
closes that gap: it builds a *correct* DF17 extended-squitter frame (with a
real CRC-24 over generator 0xFFF409) and renders it to baseband complex IQ at
2 Msps, exactly as a HackRF / RTL-SDR would deliver it.  A simulated SDR can
then emit these frames so the 852-line Mode S decoder runs end-to-end in CI
with zero hardware.

Mode S air-interface recap (see mode-s.org, ICAO Annex 10 Vol IV):

* Sample rate is 2 Msps, so 1 us == 2 samples.
* The 8 us preamble is four pulses at sample positions [0, 2, 7, 9].
* The data field is Pulse Position Modulation (PPM) at 1 bit / us:
    - bit 1  -> energy in the FIRST half of the bit period  (high, low)
    - bit 0  -> energy in the SECOND half of the bit period (low, high)
* A long (DF17) message is 112 bits == 224 samples of data.
* The trailing 24 bits are a CRC-24 (generator 0xFFF409); for DF17 the CRC
  of the whole frame, including the CRC bytes, is zero.

The renderer produces an On/Off-Keyed (OOK) magnitude envelope on the I axis
(Q held near zero) — this is what a magnitude-based Mode S detector consumes —
and adds complex AWGN at a chosen SNR.  Pure NumPy; CommPy (BSD-3) is not
required.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# Air-interface constants (kept in sync with the addon Mode S decoder)
# ---------------------------------------------------------------------------

ADSB_SAMPLE_RATE = 2_000_000        # 2 Msps
SAMPLES_PER_US = 2                  # 1 us == 2 samples at 2 Msps
PREAMBLE_SAMPLES = 16              # 8 us preamble
PREAMBLE_PULSE_POSITIONS = [0, 2, 7, 9]
LONG_MSG_BITS = 112               # DF17 extended squitter
SHORT_MSG_BITS = 56

# CRC-24 generator polynomial for Mode S (ICAO Annex 10 Vol IV).
CRC24_GENERATOR = 0xFFF409

# Downlink Format for ADS-B extended squitter.
DF_EXTENDED_SQUITTER = 17

_FrameLike = Union[bytes, bytearray, Sequence[int]]


# ---------------------------------------------------------------------------
# CRC-24 and frame construction
# ---------------------------------------------------------------------------

def crc24(data: bytes | bytearray | Sequence[int]) -> int:
    """Compute the Mode S CRC-24 over ``data`` (generator 0xFFF409).

    Args:
        data: Message bytes WITHOUT the trailing 3 CRC bytes (11 for DF17).

    Returns:
        The 24-bit CRC value as an int.
    """
    crc = 0
    for byte in data:
        crc ^= (int(byte) & 0xFF) << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= CRC24_GENERATOR
    return crc & 0xFFFFFF


def build_df17_frame(
    icao: int,
    type_code: int,
    payload: bytes | bytearray | Sequence[int] = b"\x00" * 6,
    capability: int = 5,
) -> bytes:
    """Build a complete 14-byte DF17 ADS-B frame with a valid CRC-24.

    Layout (14 bytes / 112 bits):

    ====  =========================================================
    byte  contents
    ====  =========================================================
    0     DF (5 bits, =17) | CA capability (3 bits)
    1-3   ICAO 24-bit address
    4     ME field byte 0: type code (5 bits) | 3 low ME bits (=0)
    5-10  ME field bytes 1-6 (the 6-byte ``payload``)
    11-13 CRC-24 over bytes 0..10
    ====  =========================================================

    Args:
        icao: 24-bit ICAO aircraft address.
        type_code: 5-bit ADS-B type code (1-31).
        payload: 6 bytes forming the remainder of the 7-byte ME field.
        capability: 3-bit transponder capability field (CA).

    Returns:
        14-byte frame as ``bytes`` (CRC of the whole frame is zero).
    """
    payload = bytes(payload)
    if len(payload) != 6:
        raise ValueError(f"payload must be exactly 6 bytes, got {len(payload)}")
    if not 0 <= icao <= 0xFFFFFF:
        raise ValueError("icao must fit in 24 bits")
    if not 0 <= type_code <= 0x1F:
        raise ValueError("type_code must fit in 5 bits")

    data = bytearray(11)
    data[0] = ((DF_EXTENDED_SQUITTER & 0x1F) << 3) | (capability & 0x07)
    data[1] = (icao >> 16) & 0xFF
    data[2] = (icao >> 8) & 0xFF
    data[3] = icao & 0xFF
    data[4] = (type_code & 0x1F) << 3   # 3 low ME bits left zero
    data[5:11] = payload

    crc = crc24(data)
    frame = bytearray(data)
    frame.append((crc >> 16) & 0xFF)
    frame.append((crc >> 8) & 0xFF)
    frame.append(crc & 0xFF)
    return bytes(frame)


# ---------------------------------------------------------------------------
# Baseband IQ rendering
# ---------------------------------------------------------------------------

def _frame_to_bits(frame: bytes) -> list[int]:
    """Expand a frame to its MSB-first bit list."""
    bits: list[int] = []
    for byte in frame:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _render_one(frame: bytes, amplitude: float) -> np.ndarray:
    """Render a single frame to an OOK magnitude envelope (real, float32).

    Produces ``PREAMBLE_SAMPLES + n_bits * SAMPLES_PER_US`` samples.
    """
    bits = _frame_to_bits(frame)
    n = PREAMBLE_SAMPLES + len(bits) * SAMPLES_PER_US
    env = np.zeros(n, dtype=np.float32)

    # Preamble pulses (each pulse is 0.5 us == 1 sample high at 2 Msps).
    for p in PREAMBLE_PULSE_POSITIONS:
        env[p] = amplitude

    # PPM data: bit 1 -> (high, low), bit 0 -> (low, high).
    base = PREAMBLE_SAMPLES
    for i, bit in enumerate(bits):
        idx = base + i * SAMPLES_PER_US
        if bit:
            env[idx] = amplitude       # first half high
        else:
            env[idx + 1] = amplitude   # second half high
    return env


def synth_modes_iq(
    frames: _FrameLike | Iterable[_FrameLike],
    snr_db: float = 30.0,
    amplitude: float = 1.0,
    gap_samples: int = 32,
    lead_samples: int = 8,
    seed: int | None = None,
) -> np.ndarray:
    """Render one or more Mode S frames to baseband complex IQ.

    The envelope is placed on the I axis (OOK), Q held at zero, then complex
    AWGN is added at ``snr_db`` relative to the pulse amplitude.  Output is
    interleavable / consumable by any magnitude-based Mode S decoder.

    Args:
        frames: A single 14-byte frame (bytes / sequence of ints) OR an
            iterable of such frames.
        snr_db: Signal-to-noise ratio of the pulses vs. AWGN, in dB.
        amplitude: Peak pulse amplitude (linear) on the I axis.
        gap_samples: Quiet samples inserted between consecutive frames.
        lead_samples: Quiet samples before the first frame.
        seed: Optional RNG seed for reproducible noise.

    Returns:
        1-D ``np.complex64`` array of baseband IQ samples.
    """
    # Normalize input to a list of byte frames.
    frame_list: list[bytes]
    if isinstance(frames, (bytes, bytearray)):
        frame_list = [bytes(frames)]
    else:
        seq = list(frames)
        if seq and isinstance(seq[0], int):
            # A bare sequence of ints == a single frame.
            frame_list = [bytes(seq)]  # type: ignore[arg-type]
        else:
            frame_list = [bytes(f) for f in seq]

    rng = np.random.default_rng(seed)

    parts: list[np.ndarray] = []
    if lead_samples > 0:
        parts.append(np.zeros(lead_samples, dtype=np.float32))
    for k, frame in enumerate(frame_list):
        if k > 0 and gap_samples > 0:
            parts.append(np.zeros(gap_samples, dtype=np.float32))
        parts.append(_render_one(frame, amplitude))
    if gap_samples > 0:
        parts.append(np.zeros(gap_samples, dtype=np.float32))

    env = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    # Complex baseband: signal on I, quiet Q.
    iq = env.astype(np.complex64)

    # Add complex AWGN at the requested SNR (relative to pulse power).
    if snr_db is not None and np.isfinite(snr_db):
        signal_power = float(amplitude) ** 2
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        sigma = np.sqrt(noise_power / 2.0)  # split across I and Q
        noise = (rng.standard_normal(env.shape) + 1j * rng.standard_normal(env.shape))
        iq = iq + (sigma * noise).astype(np.complex64)

    return iq.astype(np.complex64)

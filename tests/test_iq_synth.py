# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for the synthetic-IQ generator and SimulatedSDR.read_iq() seam.

RF is sensor priority #1 but untestable without a radio.  These tests prove
that the Mode S / ADS-B math (preamble correlation, 112-bit PPM bit-slice,
CRC-24 over generator 0xFFF409, ICAO / type-code round-trip) can be exercised
in CI with zero hardware, by synthesizing baseband IQ and decoding it back
with an in-test minimal Mode S decoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from tritium_lib.sdr import SimulatedSDR
from tritium_lib.sdr.iq_synth import (
    CRC24_GENERATOR,
    PREAMBLE_PULSE_POSITIONS,
    SAMPLES_PER_US,
    build_df17_frame,
    crc24,
    synth_modes_iq,
)


# ---------------------------------------------------------------------------
# In-test minimal Mode S decoder (independent of the addon, mirrors mode-s.org)
# ---------------------------------------------------------------------------

PREAMBLE_SAMPLES = 16
LONG_MSG_BITS = 112


def _crc24_check(msg_bytes: bytes) -> bool:
    """Independent CRC-24 verification: CRC over whole frame must be zero."""
    crc = 0
    for byte in msg_bytes:
        crc ^= byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= CRC24_GENERATOR
    return (crc & 0xFFFFFF) == 0


def _check_preamble(mag: np.ndarray, pos: int, threshold: float) -> bool:
    if pos + PREAMBLE_SAMPLES >= len(mag):
        return False
    for p in PREAMBLE_PULSE_POSITIONS:
        if mag[pos + p] < threshold:
            return False
    return True


def _extract_bytes(mag: np.ndarray, start: int, num_bits: int) -> bytes:
    bits = []
    for b in range(num_bits):
        idx = start + b * SAMPLES_PER_US
        first_half = float(mag[idx])
        second_half = float(mag[idx + 1])
        bits.append(1 if first_half > second_half else 0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for j in range(8):
            v = (v << 1) | bits[i + j]
        out.append(v)
    return bytes(out)


def _decode_first_frame(iq: np.ndarray) -> bytes | None:
    """Correlate preamble, slice 112 bits, return the 14 message bytes."""
    mag = np.abs(iq).astype(np.float32)
    n = len(mag)
    min_len = PREAMBLE_SAMPLES + LONG_MSG_BITS * SAMPLES_PER_US
    noise_floor = float(np.median(mag))
    threshold = max(noise_floor * 2.0, 0.1)
    i = 0
    while i < n - min_len:
        if mag[i] < threshold or not _check_preamble(mag, i, threshold):
            i += 1
            continue
        msg_start = i + PREAMBLE_SAMPLES
        return _extract_bytes(mag, msg_start, LONG_MSG_BITS)
    return None


# ---------------------------------------------------------------------------
# crc24 / frame builder
# ---------------------------------------------------------------------------

def test_crc24_generator_constant():
    assert CRC24_GENERATOR == 0xFFF409


def test_build_df17_frame_is_14_bytes_with_valid_crc():
    frame = build_df17_frame(icao=0xABCDEF, type_code=11, payload=bytes(6))
    assert isinstance(frame, (bytes, bytearray))
    assert len(frame) == 14
    # CRC over the whole 14-byte frame (incl. 3 CRC bytes) must be zero.
    assert _crc24_check(frame)


def test_build_df17_frame_fields_roundtrip():
    icao = 0x4840D6
    tc = 4
    payload = bytes([0x10, 0x20, 0x30, 0x40, 0x50, 0x60])
    frame = build_df17_frame(icao=icao, type_code=tc, payload=payload)
    df = (frame[0] >> 3) & 0x1F
    got_icao = (frame[1] << 16) | (frame[2] << 8) | frame[3]
    got_tc = (frame[4] >> 3) & 0x1F
    assert df == 17
    assert got_icao == icao
    assert got_tc == tc
    # ME field (bytes 4..10) low bits of byte4 + payload bytes preserved
    assert frame[5:11] == payload


def test_crc24_helper_matches_independent_check():
    frame = build_df17_frame(icao=0x010203, type_code=17, payload=bytes(6))
    data = frame[:11]
    transmitted = (frame[11] << 16) | (frame[12] << 8) | frame[13]
    assert crc24(data) == transmitted


# ---------------------------------------------------------------------------
# synth_modes_iq -> in-test decoder round-trip
# ---------------------------------------------------------------------------

def test_synth_iq_decodes_back_to_valid_crc_and_fields():
    icao = 0x3C6589
    tc = 11
    payload = bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x00])
    frame = build_df17_frame(icao=icao, type_code=tc, payload=payload)

    iq = synth_modes_iq(frame, snr_db=30.0, gap_samples=40, seed=7)
    assert iq.dtype == np.complex64
    assert iq.ndim == 1

    msg = _decode_first_frame(iq)
    assert msg is not None, "preamble not found / no frame decoded"
    assert len(msg) == 14
    assert _crc24_check(msg), "decoded frame failed CRC-24"

    df = (msg[0] >> 3) & 0x1F
    got_icao = (msg[1] << 16) | (msg[2] << 8) | msg[3]
    got_tc = (msg[4] >> 3) & 0x1F
    assert df == 17
    assert got_icao == icao
    assert got_tc == tc


def test_synth_iq_survives_noise():
    """Even with moderate AWGN the high-SNR pulses should slice cleanly."""
    frame = build_df17_frame(icao=0x111111, type_code=1, payload=bytes(6))
    iq = synth_modes_iq(frame, snr_db=20.0, seed=1)
    msg = _decode_first_frame(iq)
    assert msg is not None
    assert _crc24_check(msg)


def test_synth_multiple_frames():
    f1 = build_df17_frame(icao=0xAAAAAA, type_code=11, payload=bytes(6))
    f2 = build_df17_frame(icao=0xBBBBBB, type_code=4, payload=bytes(6))
    iq = synth_modes_iq([f1, f2], snr_db=30.0, seed=3)
    # Long enough to hold two preamble+message blocks plus gaps.
    one = PREAMBLE_SAMPLES + LONG_MSG_BITS * SAMPLES_PER_US
    assert len(iq) >= 2 * one


# ---------------------------------------------------------------------------
# SimulatedSDR.read_iq() seam
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_simulated_sdr_read_iq_shape_dtype():
    sdr = SimulatedSDR(seed=42)
    await sdr.detect()
    await sdr.tune(1_090_000_000, sample_rate=2_000_000)
    samples = await sdr.read_iq(8192)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (8192,)
    assert samples.dtype == np.complex64


@pytest.mark.asyncio
async def test_simulated_sdr_read_iq_contains_decodable_adsb_when_tuned_1090():
    """When tuned to 1090 MHz the simulator emits decodable ADS-B frames."""
    sdr = SimulatedSDR(seed=99)
    await sdr.detect()
    await sdr.tune(1_090_000_000, sample_rate=2_000_000)
    iq = await sdr.read_iq(60_000)
    msg = _decode_first_frame(iq)
    assert msg is not None, "no ADS-B frame in 1090 MHz IQ"
    assert _crc24_check(msg)
    df = (msg[0] >> 3) & 0x1F
    assert df == 17


def test_base_read_iq_default_not_implemented():
    """The ABC default read_iq raises NotImplementedError (sync probe)."""
    from tritium_lib.sdr.base import SDRDevice

    # A minimal concrete subclass that does NOT override read_iq.
    class _Bare(SDRDevice):
        async def detect(self):  # pragma: no cover - trivial
            return None

        async def sweep(self, a, b, bin_width_hz=500000):  # pragma: no cover
            return None

        async def tune(self, freq_hz, sample_rate=2000000, bandwidth=0):  # pragma: no cover
            return None

        async def stop(self):  # pragma: no cover
            return None

    bare = _Bare()
    import asyncio

    with pytest.raises(NotImplementedError):
        asyncio.run(bare.read_iq(16))


# ---------------------------------------------------------------------------
# OPTIONAL: cross-check against the real addon decoder if importable.
# ---------------------------------------------------------------------------

def test_optional_real_addon_decoder_validates_crc():
    """Best-effort: feed the addon ADS-B decoder our synth CRC helper.

    The addon decoder operates on int8 magnitude, so we only cross-check the
    CRC math here (shared generator).  Skipped cleanly if not importable.
    """
    import importlib
    import sys

    addon_path = "/home/scubasonar/Code/tritium/tritium-addons/hackrf"
    if addon_path not in sys.path:
        sys.path.insert(0, addon_path)
    try:
        adsb = importlib.import_module("hackrf_addon.decoders.adsb")
    except Exception:
        pytest.skip("hackrf addon ADS-B decoder not importable")

    frame = build_df17_frame(icao=0x123456, type_code=11, payload=bytes(6))
    # Our generator's CRC == addon's crc24 over the same data.
    assert crc24(frame[:11]) == adsb.crc24(frame[:11])
    # And the addon's whole-frame validator accepts our frame.
    assert adsb.validate_crc(bytes(frame))

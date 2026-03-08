# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Acoustic modem models — protocol types for data-over-audio communication.

These models define the framing, configuration, and channel statistics for
acoustic data links between devices.  Useful for short-range comms where
RF is unavailable or undesirable (e.g., underwater, EMI-restricted zones).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModulationType(str, Enum):
    """Supported acoustic modulation schemes."""
    FSK = "fsk"
    PSK = "psk"
    OFDM = "ofdm"


class AcousticFrame(BaseModel):
    """A single acoustic data frame.

    Mirrors the on-air framing: preamble for sync detection, payload,
    CRC for integrity, and modulation metadata.
    """
    preamble: bytes = b""
    payload_bytes: bytes = b""
    crc: int = 0
    modulation: ModulationType = ModulationType.FSK
    frame_id: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def payload_size(self) -> int:
        return len(self.payload_bytes)

    @property
    def total_size(self) -> int:
        """Total frame size including preamble."""
        return len(self.preamble) + len(self.payload_bytes)


class AcousticConfig(BaseModel):
    """Configuration for an acoustic modem channel.

    Defines the carrier frequency, baud rate, and modulation scheme
    for a data-over-audio link.
    """
    frequency_hz: int = 1000  # carrier frequency in Hz
    baud_rate: int = 300  # symbols per second
    modulation: ModulationType = ModulationType.FSK
    preamble_ms: int = 100  # preamble duration in milliseconds
    sample_rate_hz: int = 44100
    bits_per_symbol: int = 1


class AcousticChannelStats(BaseModel):
    """Statistics for an acoustic communication channel.

    Collected over a measurement window to assess link quality.
    """
    snr_db: float = 0.0  # signal-to-noise ratio in dB
    bit_error_rate: float = 0.0  # 0.0 = perfect, 1.0 = all errors
    throughput_bps: float = 0.0  # effective throughput in bits per second
    frames_sent: int = 0
    frames_received: int = 0
    frames_dropped: int = 0
    retransmissions: int = 0

    @property
    def frame_loss_rate(self) -> float:
        """Fraction of frames lost (0.0 to 1.0)."""
        total = self.frames_received + self.frames_dropped
        if total == 0:
            return 0.0
        return self.frames_dropped / total

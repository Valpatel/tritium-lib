# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""SDR (Software Defined Radio) abstractions for Tritium.

Provides a generic SDRDevice base class that specific hardware implementations
(HackRF, RTL-SDR, LimeSDR, etc.) extend. Each device can:
- Detect and enumerate hardware
- Tune to a frequency
- Capture IQ samples
- Run broadband sweeps
- Report device status

Architecture mirrors the firmware flasher pattern:
    SDRDevice (ABC)
    ├── HackRFDevice — hackrf_* CLI tools
    ├── RTLSDRDevice — rtl_* CLI tools (future)
    └── LimeSDRDevice — LimeSuite tools (future)
"""

from .base import SDRDevice, SDRInfo, SweepResult, SweepPoint

__all__ = ["SDRDevice", "SDRInfo", "SweepResult", "SweepPoint"]

# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Perception pipeline — frame analysis and conversational fact extraction.

L0: Quality gate (sharpness, brightness)
L1: Complexity (edge density)
L2: Motion (frame diff)
Plus: LLM chat API client, regex fact extraction from conversation.

Framework-free: pure OpenCV/numpy/stdlib. The LLM host is injected via
set_ollama_host(); the PTZ pose estimator accepts any PTZPosition.
"""

from tritium_lib.perception.perception import (
    CameraPose,
    FrameAnalyzer,
    FrameMetrics,
    PoseEstimator,
    PTZPosition,
)
from tritium_lib.perception.extraction import extract_facts, extract_person_name
from tritium_lib.perception.vision import (
    check_radio_detection,
    ollama_chat,
    set_ollama_host,
)

__all__ = [
    "CameraPose",
    "FrameAnalyzer",
    "FrameMetrics",
    "PoseEstimator",
    "PTZPosition",
    "check_radio_detection",
    "extract_facts",
    "extract_person_name",
    "ollama_chat",
    "set_ollama_host",
]
